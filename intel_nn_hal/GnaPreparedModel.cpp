#define LOG_TAG "GnaPreparedModel"

#include "GnaPreparedModel.h"
#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>
#include <fstream>
#include <thread>
#include <time.h>
#include <chrono>
#include <xmmintrin.h>
#include <immintrin.h>
#include "Dequantize.h"
#include "Quantize.h"
#include "EmbeddingLookup.h"

using namespace std::chrono;

#include "ValidateHal.h"
#include "Utils.h"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
typedef std::chrono::duration<float> fsec;

//#include "ExecutionBurstServer.h"
//#include "OperationsUtils.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

using namespace android::nn;
using namespace IRBuilder::LstmLayer;
static const Timing kNoTiming = {.timeOnDevice = UINT64_MAX, .timeInDriver = UINT64_MAX};

template <typename T>
T getOperandConstVal(const Model& model, const Operand& operand) {
    const T* data = reinterpret_cast<const T*>(&model.operandValues[operand.location.offset]);
    return data[0];
}

// TODO: Code duplication
static Return<void> notify(const sp<V1_0::IExecutionCallback>& callback, const V1_0_ErrorStatus& status,
                           const hidl_vec<OutputShape>&, Timing) {
    return callback->notify(status);
}

#ifdef CACHING
inline void writeNBytes(const void *ptr, uint32_t size, int fd) {
    auto ret = write(fd, static_cast<const char*>(ptr), size);
}

template <class T>
inline void writeBits(const T & obj, int fd) {
    auto ret = write(fd, reinterpret_cast<const char *>(&obj), sizeof(T));
}

template <class T>
inline void readBits(T & obj, int fd) {
    auto ret = read(fd, reinterpret_cast<char *>(&obj), sizeof(T));
}

inline void readNBytes(void * ptr, uint32_t size, int fd) {
    auto ret = read(fd, reinterpret_cast<char *>(ptr), size);
}

template <int nBits, class T>
inline void readNBits(T & obj, int fd) {
    std::array<uint8_t, nBits / 8> tmp;
    auto ret= read(fd, reinterpret_cast<char *>(&tmp), nBits / 8);
    obj = * reinterpret_cast<T*>(&tmp.front());
}
#endif

void GnaPreparedModel::initializeInput() {
    /* copy weights, biases ..etc */
    for (auto i=0; i < mModel.main.inputIndexes.size(); i++) {
        auto curIndex = mModel.main.inputIndexes[i];
        mModelInputIndices.emplace_back(mModel.main.inputIndexes[i]);
        // ALOGE("Searching for input index:%d", curIndex);

        auto itr = std::find_if(mInputPorts.begin(), mInputPorts.end(),
                                    [&](const std::pair<uint32_t, LayerInfo>& elem) {
                                        return (elem.first == curIndex);
                                    });
        if (itr != mInputPorts.end()) {
            mlayerInputIndices.push_back(i);
        } else {
            nnAssert(false);
        }
    }

    for (auto i =0; i < mModel.main.outputIndexes.size(); i++) {
        mModelOutputIndices.emplace_back(mModel.main.outputIndexes[i]);
    }
}

void GnaPreparedModel::initializeInput(std::vector<uint32_t>& indexVec) {
    /* copy weights, biases ..etc */
    for (auto i=0; i < indexVec.size(); i++) {
        auto curIndex = indexVec[i];
        // ALOGE("Searching for input index:%d", curIndex);

        auto itr = std::find_if(mInputPorts.begin(), mInputPorts.end(),
                                    [&](const std::pair<uint32_t, LayerInfo>& elem) {
                                        return (elem.first == curIndex);
                                    });
        if (itr != mInputPorts.end()) {
            mlayerInputIndices.push_back(i);
        } else {
            nnAssert("Cannot set data to non-input layers during infer request");
        }
    }
}

std::tuple<ErrorStatus, hidl_vec<V1_2::OutputShape>, Timing>
GnaPreparedModel::executeSynchronouslyBase(const V1_3::Request& request, V1_2::MeasureTiming measure,
                  //BasePreparedModel* preparedModel,
                  const V1_3::OptionalTimePoint& halDeadline,
                  const V1_3::OptionalTimeoutDuration& loopTimeoutDuration) {
    ALOGD("%s", __func__);

    time_point driverStart;
    if (measure == MeasureTiming::YES) driverStart = now();

    return syncExecute(request, measure, driverStart);
}

bool GnaPreparedModel::finalizeOutput(/*RunTimeOperandInfo* output */) {
    for (auto i : mModel.main.outputIndexes) {
        int dims_size = mOperands[i].dimensions.size();

        //VLOG(L1, "mPorts[%d] %s dims size %d", i, mPorts[i]->getName().c_str(), dims_size);
        VLOGDIMS(L1, mOperands[i].dimensions, "current operand Output dims:");
        /* VLOGDIMS(L1, mPorts[i]->getTensorDesc().getDims(), "Real Output dims:");

        auto outputDims = mPorts[i]->getTensorDesc().getDims();

        uint32_t nelem = getNumberOfElements(mOperands[i].dimensions);
        auto outputElem = sizeOfTensor(outputDims);
        if (nelem != outputElem) {
            VLOG(L1, "set correct dims as operand output dims different than real output dims\n");
        } */
    }
    return true;
}

bool GnaPreparedModel::constructGNAGraph(std::pair<int, int> indices) {
    bool success = false;
    mBuilderModel = new IRBuilder::ModelBuilder();
    mBuilderModel->initializeBuilder();
    gnaPluginPtr = new GnaNetwork();

    time_point irbuild_start = now();
    for (size_t i=std::get<0>(indices); i <= std::get<1>(indices); ++i) {
        auto operation = mModel.main.operations[i];
        dumpOperation(operation);
        switch (operation.type) {
           case OperationType::QUANTIZED_LSTM:
                success = operationQuantizedLSTM(operation);
                break;
           case OperationType::LSTM:
                success = operationLSTM(operation);
                break;
            case OperationType::FULLY_CONNECTED:
                success = operationFullyConnected(operation);
                break;
            case OperationType::ADD:
                success = operationAdd(operation);
                break;
            case OperationType::TANH:
                success = operationTANH(operation);
                break;
            default:
                return false;
        }

        if (success == false) {
            return false;
        }
    }
    auto network = mBuilderModel->convertBuilder();
#ifdef PERF_COUNTERS
    time_point irbuild_end = now();
    runtimeMetrics.irBuild_time = (double(millisecondsDuration(irbuild_end, irbuild_start)));
    gnaPluginPtr->setNetwork(network);
    InferenceEngine::CNNNetwork passed_network({network});
    gnaPluginPtr->loadNetwork(passed_network, isDecoderNw);
    time_point gnabuild_end = now();
    runtimeMetrics.nw_load_time = (double(millisecondsDuration(gnabuild_end, irbuild_end)));
#else
    //gnaPluginPtr = new GnaNetwork(network, "GNA"); // Why are we doing this??
    gnaPluginPtr->setNetwork(network);
    InferenceEngine::CNNNetwork passed_network({network});
    gnaPluginPtr->loadNetwork(passed_network, isDecoderNw);
#endif
    for (auto item:mModelIRBlobs) {
        item->deallocate();
    }
    gnaPluginPtr->queryState();
    gnaPluginPtr->reset();

#ifdef CACHING
    if (modelCache.size() > 0 ) {
        // Export the graph to cache
        auto modelFd = modelCache[0]->data[0];
        gnaPluginPtr->exportGraph("NNCACHE" + std::to_string(modelFd));

        // TODO: Check for file integrity
        auto dataCacheFd = modelCache[1]->data[0];
        auto operandCount = mModel.main.operands.size();
        writeBits(operandCount, dataCacheFd);

        for (auto i=0; i < operandCount; i++) {
            RunTimeOperandInfo& runtimeOp = mOperands[i];

            int type = static_cast<int>(runtimeOp.type);
            writeBits(type, dataCacheFd);

            auto sizeOfVec = runtimeOp.dimensions.size();
            writeBits(sizeOfVec, dataCacheFd);

            for (auto val: runtimeOp.dimensions)
                writeBits(val, dataCacheFd);

            // TODO: For float is this best way to serialize the value???
            writeBits(runtimeOp.scale, dataCacheFd);
            writeBits(runtimeOp.zeroPoint, dataCacheFd);
            writeBits(runtimeOp.lifetime, dataCacheFd);
            writeBits(runtimeOp.numberOfUsesLeft, dataCacheFd);
        }

        // Write input indexes and output indexes
        auto ioIndexSize = mModel.main.inputIndexes.size();
        writeBits(ioIndexSize, dataCacheFd);
        for (auto i=0; i < mModel.main.inputIndexes.size(); i++) {
            auto index = mModel.main.inputIndexes[i];
            writeBits(index, dataCacheFd);
        }

        ioIndexSize = mModel.main.outputIndexes.size();
        writeBits(ioIndexSize, dataCacheFd);
        for (auto i=0; i < mModel.main.outputIndexes.size(); i++) {
            auto index = mModel.main.outputIndexes[i];
            writeBits(index, dataCacheFd);
        }

        auto inputSize = mInputPorts.size();
        writeBits(inputSize, dataCacheFd);
        for (auto iter: mInputPorts) {
            // index
            writeBits(iter.first, dataCacheFd);

            // string
            writeBits(static_cast<uint32_t>(sizeof(iter.second.layerName.size() + 1)), dataCacheFd);
            writeBits(strlen(iter.second.layerName.c_str()) + 1, dataCacheFd);
            writeNBytes(iter.second.layerName.c_str(), strlen(iter.second.layerName.c_str()) + 1 , dataCacheFd);

            // bool
            int memoryLayer = iter.second.memoryLayer?1:0;
            writeBits(memoryLayer, dataCacheFd);
        }

        // Store the output names
        inputSize = mOutputToLayerMap.size();
        writeBits(inputSize, dataCacheFd);
        for (auto iter: mOutputToLayerMap) {
            // index
            writeBits(iter.first, dataCacheFd);

            // string
            writeBits(static_cast<uint32_t>(sizeof(iter.second.layerName.size() + 1)), dataCacheFd);
            writeBits(strlen(iter.second.layerName.c_str()) + 1, dataCacheFd);
            writeNBytes(iter.second.layerName.c_str(),
                        strlen(iter.second.layerName.c_str()) + 1 , dataCacheFd);
        }

        // TODO: Identify location to store the hash value
        dataCacheFd = modelCache[2]->data[0];
        std::string hashStr = computeHashFromFd(modelFd);
        const auto hashLen = strlen(hashStr.c_str());
        writeBits(static_cast<uint32_t>(hashLen), dataCacheFd);
        writeNBytes(hashStr.c_str(), hashLen, dataCacheFd);

        // TODO: Remove this once model details can be obtained from application
        writeBits(static_cast<uint32_t>(modelNameStr.length()), dataCacheFd);
        writeNBytes(modelNameStr.c_str(), hashLen, dataCacheFd);
    }
#endif
    for (auto runtimeInfo : mPoolInfos) {
        runtimeInfo.update();
    }
    for (auto runtimeInfo : mPoolInfos) {
        runtimeInfo.unmap_mem();
    }

    return true;
}

OpContainer* GnaPreparedModel::constructCpuGraph(std::pair<int, int> indices) {
    OpContainer* subgraphOps = new OpContainer(true);

    VLOG(L1, "Index start:%d end:%d", std::get<0>(indices), std::get<1>(indices));
    for (size_t i=std::get<0>(indices); i <= std::get<1>(indices); ++i) {
        auto operation = mModel.main.operations[i];
        BaseOp* cpuOperation = nullptr;
        bool success = false;

        dumpOperation(operation);
        switch (operation.type) {
           case OperationType::DEQUANTIZE:
                if(isRnnT())
                    cpuOperation = operationDequantize(operation, true);
                else
                    cpuOperation = operationDequantize(operation, false);
                if (!cpuOperation) {
                    VLOG(L1, "Failed to create dequantize operation !!!!");
                }
                subgraphOps->addOperation(cpuOperation);
                success = true;
                break;

            case OperationType::QUANTIZE:
                if (isRnnT()) {
			        if (isJointNw) {
                        cpuOperation = operationQuantize(operation, false);
                    }
                    else
                        cpuOperation = operationQuantize(operation, true);
                }
                else {
                    cpuOperation = operationQuantize(operation, false);
                }
                if (!cpuOperation) {
                    VLOG(L1, "Failed to create Quantize operation !!!!");
                }
                subgraphOps->addOperation(cpuOperation);
                success = true;
                break;
            case OperationType::EMBEDDING_LOOKUP:
                cpuOperation = operationEmbeddingLookup(operation);
                if (!cpuOperation) {
                    VLOG(L1, "Failed to create Embedding Lookup operation !!!!");
                }
                subgraphOps->addOperation(cpuOperation);
                success = true;
                break;
            default:
                VLOG(L1, "unsupported operation %d", operation.type);
                break;
        }

        if (success == false) {
            delete subgraphOps;
            VLOG(L1, "failed to convert operation %d", operation.type);
            return nullptr;
        }
    }

    return subgraphOps;
}

BaseOp* GnaPreparedModel::getCpuOpFromLayerName(std::string layer) {
    BaseOp* ptrOp = nullptr;
    for (auto container: mNwManager) {
        if (container->isCpuGraph()) {
            ptrOp = container->getCpuOpFromLayerName(layer);
            if (ptrOp)
                return ptrOp;
        }
    }

    // if (!ptrOp)
    //     VLOG(L1, "Failed to find cpu layer for layer name: %s", layer.c_str());

    return ptrOp;
}

bool GnaPreparedModel::initialize(const hidl_vec<hidl_handle>& modelCache, const HidlToken& token) {
    bool success = false;

    // TODO: Remove this hack to identify the nw based on operations
    int lstmCount = 0;
    bool hasFC = false;
    int  quantizeCnt = 0;

    for (auto op: mModel.main.operations) {
        if (op.type == OperationType::FULLY_CONNECTED) {
            hasFC = true;
        } else if (op.type == OperationType::QUANTIZED_LSTM) {
            lstmCount++;
        } else if (op.type == OperationType::QUANTIZE) {
            quantizeCnt++;
        }
    }

    if (!hasFC && quantizeCnt != 2 && lstmCount == 2) {
        isEnc0Nw = true;
        modelNameStr = "Encoder0";
    }
    else if (quantizeCnt == 2 && hasFC) {
        VLOG(L1, "initialize quantize %d", quantizeCnt);
        isJointNw = true;
        modelNameStr = "Joint";
    }
    else {
        if (lstmCount > 3) {
            isEnc1Nw = true;
            modelNameStr = "Encoder1";
        } else if (lstmCount == 2){
            isDecoderNw = true;
            modelNameStr = "Decoder";
        }
    }

    // TODO Check operation supoorted or not, user may not call getOpertionSupported()
    for (const auto& operation : mModel.main.operations) {
        success = isOperationSupported(operation, mModel, mTargetDevice);
        dumpOperationSupport(operation, success);
        if (!success) {
            VLOG(L1, "get unsupported operation in initialize()");
            return false;
        }
    }

    success = setRunTimePoolInfosFromHidlMemories_1_3(&mPoolInfos, mModel.pools);
    if (!success) {
        VLOG(L1, "setRunTimePoolInfosFromHidlMemories failed.\n");
        return false;
    }

    success = initializeRunTimeOperandInfo();
    if (!success) {
        VLOG(L1, "initializeRunTimeOperandInfo failed.\n");
        return false;
    }

    auto isCpuOp = [](const Operation& op) -> bool {
        bool success = false;
        switch(op.type) {
            case OperationType::DEQUANTIZE:
                success = true;
                break;
            case OperationType::QUANTIZE:
                success = true;
                break;
            case OperationType::EMBEDDING_LOOKUP:
                success = true;
                break;
            default:
                break;
        }

        return success;
    };

    // Calculate the indices of subgraphs
    std::vector<std::pair<int,int>> subgraphs;
    std::vector<bool> cpuTarget;
    int startIndex = 0;
    bool prevOpCpu = false;

    for (size_t i =0; i < mModel.main.operations.size(); i++) {
        auto op = mModel.main.operations[i];
        if (i == 0) {
            prevOpCpu = isCpuOp(op);
            continue;
        } else {
            bool curOpOnCpu = isCpuOp(op);

            if (curOpOnCpu != prevOpCpu) {
                subgraphs.push_back(std::make_pair(startIndex, i - 1));
                if (prevOpCpu) {
                    cpuTarget.push_back(true);
                } else {
                    cpuTarget.push_back(false);
                }
                startIndex = i;
            }
            prevOpCpu = curOpOnCpu;
        }
    }
    subgraphs.push_back(std::make_pair(startIndex, mModel.main.operations.size()-1));
    if (prevOpCpu) {
        cpuTarget.push_back(true);
    } else {
        cpuTarget.push_back(false);
    }

    // Check how many graphs run on GNA
    int gnaGraphcount = 0;
    for(auto i : cpuTarget) {
        if (i == false)
            gnaGraphcount++;
    }

    if (gnaGraphcount > 1) {
        VLOG(L1,  "Can not delegate more than 1 graph on GNA currently in single driver instance");
        nnAssert(false);
    }

    for (auto int i=0; i < cpuTarget.size(); i++) {
        std::pair<int, int> indexRange = subgraphs[i];

        if (cpuTarget[i] == true) {
            // CPU target
            OpContainer* opsContainer = constructCpuGraph(indexRange);
            if (opsContainer){
                mNwManager.push_back(opsContainer);
            }
        } else {
            if (constructGNAGraph(indexRange)) {
                OpContainer* opsContainer = new OpContainer(false);
                opsContainer->addOperation(gnaPluginPtr);
                mNwManager.push_back(opsContainer);
            }
            else {
                VLOG(L1,  " could not init container\n");
                return false;
            }
        }
    }

    initializeInput();

    return true;
}

#ifdef CACHING
bool GnaPreparedModel::initializeFromCache(const hidl_vec<hidl_handle>& modelCache, const HidlToken& token) {
    time_point irBuildStart = now();
    // Load the network from cache file
    gnaPluginPtr = new GnaNetwork(nullptr, "GNA");
    gnaPluginPtr->importNetwork("NNCACHE" + std::to_string(modelCache[0]->data[0]), isDecoderNw);
    gnaPluginPtr->queryState();
    gnaPluginPtr->reset();

    std::string hash = computeHashFromFd(modelCache[0]->data[0]);

    // Read the Hash value
    auto dataCacheFd = modelCache[2]->data[0];
    uint32_t hashLen = 0, modelStrLen = 0;

    readNBits<32>(hashLen, dataCacheFd);
    std::string storedHashString("", hashLen);
    readNBytes(&storedHashString[0], hashLen, dataCacheFd);

    readNBits<32>(modelStrLen, dataCacheFd);
    std::string tmpModelNameString("", modelStrLen);
    readNBytes(&tmpModelNameString[0], modelStrLen, dataCacheFd);
    modelNameStr = tmpModelNameString;

    if (hash.compare(0, std::string::npos, storedHashString) != 0) {
        ALOGE("SHA512 digest does not match");
        ALOGE("Stored hash:", storedHashString.c_str());
        ALOGE("Computed hash:", hash.c_str());
        nnAssert("Model cache stored has been corrupted. Checksum does not match");
    }

    if (modelNameStr.compare(DECODER_TOKEN_STR) == 0) {
        isDecoderNw = true;
    } else if(modelNameStr.compare(ENC0_TOKEN_STR) == 0) {
        isEnc0Nw = true;
    } else if(modelNameStr.compare(ENC1_TOKEN_STR) == 0) {
        isEnc1Nw = true;
    }

    dataCacheFd = modelCache[1]->data[0];
    std::size_t operandCount = 0;
    readBits(operandCount, dataCacheFd);

    mOperands.resize(operandCount);
    for (auto i=0; i < operandCount; i++) {
        RunTimeOperandInfo& runtimeOp = mOperands[i];

        int type = 0;
        readBits(type, dataCacheFd);
        runtimeOp.type = static_cast<android::hardware::neuralnetworks::nnhal::OperandType>(type);

        std::size_t sizeOfVec = 0;
        readBits(sizeOfVec, dataCacheFd);
        for (auto i =0; i < sizeOfVec; i++) {
            uint32_t val = 0;
            readBits(val, dataCacheFd);
            runtimeOp.dimensions.emplace_back(val);
        }

        // TODO: For float is this best way to serialize the value???
        readBits(runtimeOp.scale, dataCacheFd);
        readBits(runtimeOp.zeroPoint, dataCacheFd);
        readBits(runtimeOp.lifetime, dataCacheFd);
        readBits(runtimeOp.numberOfUsesLeft, dataCacheFd);
        runtimeOp.buffer = nullptr;
        runtimeOp.length = 0;
    }

    // Write input indexes and output indexes
    std::size_t ioIndexSize = 0;
    readBits(ioIndexSize, dataCacheFd);
    for (auto i=0; i < ioIndexSize; i++) {
        auto index = 0;
        readBits(index, dataCacheFd);
        mModelInputIndices.emplace_back(index);
    }

    ioIndexSize = 0;
    readBits(ioIndexSize, dataCacheFd);
    for (auto i=0; i < ioIndexSize; i++) {
        auto index = 0;
        readBits(index, dataCacheFd);
        mModelOutputIndices.emplace_back(index);
    }

    // read size of inputs
    std::size_t vecSize = 0;
    readBits(vecSize, dataCacheFd);
    for (auto i=0; i < vecSize; i++) {
        // index
        uint32_t layerId = 0;
        readBits(layerId, dataCacheFd);

        // string
        uint32_t size = 0;
        readBits(size, dataCacheFd);
        uint32_t nameSize = 0;
        readNBits<64>(nameSize, dataCacheFd);
        std::vector<char> inName(nameSize);
        readNBytes(inName.data(), nameSize, dataCacheFd);
        std::string layerName = std::string(inName.begin(), inName.end() - 1);

        // bool
        uint32_t memLayer = 0;
        readBits(memLayer, dataCacheFd);
        mInputPorts.emplace(std::make_pair(layerId, LayerInfo(layerName, memLayer)));
    }

    // read size of ouputs
    vecSize = 0;
    readBits(vecSize, dataCacheFd);
    for (auto i=0; i < vecSize; i++) {
        // index
        uint32_t layerId = 0;
        readBits(layerId, dataCacheFd);

        // string
        uint32_t size = 0;
        readBits(size, dataCacheFd);
        uint32_t nameSize = 0;
        readNBits<64>(nameSize, dataCacheFd);
        std::vector<char> inName(nameSize);
        readNBytes(inName.data(), nameSize, dataCacheFd);
        std::string layerName = std::string(inName.begin(), inName.end() - 1);
        mOutputToLayerMap.emplace(std::make_pair(layerId, layerName));
    }

    initializeInput(mModelInputIndices);

    time_point irBuildEnd = now();
    runtimeMetrics.nw_load_time = (double(millisecondsDuration(irBuildEnd, irBuildStart)));
    return true;
}
#endif

// TODO: Call parent class deinitialize from here
void GnaPreparedModel::deinitialize() {
#ifdef PERF_COUNTERS
    VLOG(L1, "GnaPreparedModel - deinitialize");
    for (const auto &it : gnaPluginPtr->totalPerfCounters) {
               std::string const &counter_name = it.first;
               float current_units = static_cast<float>(it.second.realTime_uSec);
               float call_units = current_units / gnaPluginPtr->noInferCall;
               // if GNA HW counters
               // get frequency of GNA module
               float freq = getGnaFrequencyMHz();
               current_units /= freq * 1000;
               call_units /= freq;
              std::cout << std::setw(30) << std::left << counter_name.substr(4, counter_name.size() - 1);
              std::cout << std::setw(16) << std::right << current_units;
              std::cout << std::setw(21) << std::right << call_units;
              std::cout << std::endl;
    }
    std::vector<double>::iterator min_infer_time = std::min_element(gnaPluginPtr->inferTimeGNA.begin(), gnaPluginPtr->inferTimeGNA.end());
    // VLOG(L1, "deinitialize infer times");
    // for (auto iter: gnaPluginPtr->inferTimeGNA) {
	//         VLOG(L1, "%fms  ", iter);
    // }
    std::cout << "Minimum infer time " << gnaPluginPtr->inferTimeGNA.at(std::distance(gnaPluginPtr->inferTimeGNA.begin(), min_infer_time)) << "\n";
#endif
    if (mBuilderModel) {
        delete mBuilderModel;
        mBuilderModel = nullptr;
    }

    if (gnaPluginPtr) {
        delete gnaPluginPtr;
        gnaPluginPtr = nullptr;
    }

    VLOG(L1, "free engine");
}

#ifdef PERF_COUNTERS
#ifdef __AVX2__
bool quantizeToQuant8Signed(const float* inputData, int8_t* outputData, const Shape& outputShape,
                            metrics& runtime_metrics) {
    auto start = now();
    uint32_t len = getNumberOfElements(outputShape.dimensions);
    uint32_t moves = len >> 4;
    uint32_t mod = len % 8;
    const __m512  m_scale =  _mm512_set1_ps(outputShape.scale);
    const __m512  m_offset =  _mm512_set1_ps(outputShape.offset);

	for ( int i = 0; i < moves; i++)
    {
        __m512 x = _mm512_loadu_ps(inputData);
        __m512 div_x = _mm512_div_ps(x, m_scale);
        __m512 add_x = _mm512_add_ps(div_x, m_offset);
        __m512i add_x_i = _mm512_cvt_roundps_epi32(add_x, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        __m128i x_int =  _mm512_cvtepi32_epi8(add_x_i);
         *((__m128i *) outputData) = x_int;
        inputData += 16;
        outputData += 16;
    }

    runtime_metrics.quant_time += (double(microsecondsDuration(now(), start)));
    return true;
}

bool quantizeToQuant16(const float* inputData, uint16_t* outputData, const Shape& outputShape,
                        metrics& runtime_metrics) {
    auto start = now();
    uint32_t len = getNumberOfElements(outputShape.dimensions);
    uint32_t moves = len >> 4;
    uint32_t mod = len % 8;
    const __m512 m_scale = _mm512_set1_ps(outputShape.scale);
    const __m512 m_offset = _mm512_set1_ps(outputShape.offset);

	for ( int i = 0; i < moves; i++)
    {
        __m512 x = _mm512_loadu_ps(inputData);
        __m512 div_x = _mm512_div_ps(x, m_scale);
        __m512 add_x = _mm512_add_ps(div_x, m_offset);
        __m512i x_int = _mm512_cvtps_epi32(add_x);
        __m256i x_int_16 = _mm512_cvtepi32_epi16(x_int);
         *((__m256i *) outputData) = x_int_16;
        inputData += 16;
        outputData += 16;
    }

    runtime_metrics.quant_time += (double(microsecondsDuration(now(), start)));
    return true;
}
#elif defined(__SSE4_2__)
bool quantizeToQuant8Signed(const float* inputData, int8_t* outputData, const Shape& outputShape,
                            metrics& runtime_metrics) {
    auto start = now();
    uint32_t len = getNumberOfElements(outputShape.dimensions);
    uint32_t moves = len >> 2;
    uint32_t mod = len % 8;
    const __m128 m_scale = _mm_set1_ps(outputShape.scale);
    const __m128 m_offset = _mm_set1_ps(outputShape.offset);

	for ( int i = 0; i < moves; i++)
    {
        __m128 x = _mm_loadu_ps(inputData);
        __m128 div_x = _mm_div_ps(x, m_scale);
        __m128 add_x = _mm_add_ps(div_x, m_offset);
        __m64 x_int = _mm_cvtps_pi8(add_x);
         *((__m64 *) outputData) = x_int;
        inputData += 4;
        outputData += 4;
    }

   runtime_metrics.quant_time += (double(microsecondsDuration(now(), start)));
    return true;
}

bool quantizeToQuant16(const float* inputData, uint16_t* outputData, const Shape& outputShape,
                        metrics& runtime_metrics) {
    auto start = now();
    uint32_t len = getNumberOfElements(outputShape.dimensions);
    uint32_t moves = len >> 2;
    uint32_t mod = len % 8;
    const __m128 m_scale = _mm_set1_ps(outputShape.scale);
    const __m128 m_offset = _mm_set1_ps(outputShape.offset);

	for ( int i = 0; i < moves; i++)
    {
        __m128 x = _mm_loadu_ps(inputData);
        __m128 div_x = _mm_div_ps(x, m_scale);
        __m128 add_x = _mm_add_ps(div_x, m_offset);
        __m64 x_int = _mm_cvtps_pi16(add_x);
         *((__m64 *) outputData) = x_int;
        inputData += 4;
        outputData += 4;
    }

    runtime_metrics.quant_time += (double(microsecondsDuration(now(), start)));
    return true;
}
#endif

template<typename T>
bool deQuantize(const T* inputData, const uint32_t& len, const float scale,
                const int32_t zeroPoint, float* outputData, metrics& runtime_metrics) {
    auto start = now();
      int32_t value;
      for (int i = 0; i < len; ++i) {
        value = *(inputData + i);
        outputData[i] = static_cast<float>(scale * (value - zeroPoint));
      }
    auto end = now();
    runtime_metrics.deQuant_time += (double(millisecondsDuration(end, start)));
      return true;
}

#ifdef __AVX2__
template <>
bool deQuantize(const int8_t* inputData, const uint32_t& len, const float scale,
                const int32_t zeroPoint, float* outputData, metrics& runtime_metrics) {
    uint32_t moves = len >> 4;
    uint32_t mod = len % 8;
    const __m512 m_scale = _mm512_set1_ps(scale);
    const __m512 m_zp = _mm512_set1_ps(zeroPoint);
    auto start = now();
	for ( int i = 0; i < moves; i++)
    {
        __m128i x = _mm_load_si128((__m128i*)inputData);
        __m512i x_256 = _mm512_cvtepi8_epi32(x);
        __m512 y = _mm512_cvtepi32_ps(x_256);
        __m512 y_sub = _mm512_sub_ps(y, m_zp);
        __m512 y_mul = _mm512_mul_ps(y_sub, m_scale);
        _mm512_storeu_ps(outputData, y_mul);
	    outputData += 16;
        inputData += 16;
    }
    auto end = now();
    runtime_metrics.deQuant_time += (double(millisecondsDuration(end, start)));

    return true;
}

template <>
bool deQuantize(const uint16_t* inputData, const uint32_t& len, const float scale,
                const int32_t zeroPoint, float* outputData, metrics& runtime_metrics) {
    uint32_t moves = len >> 4;
    uint32_t mod = len % 8;
    const __m512 m_scale = _mm512_set1_ps(scale);
    auto start = now();
    for ( int i = 0; i < moves; i++)
    {
        __m256i x = _mm256_loadu_epi16(inputData);
        __m512i x_256 = _mm512_cvtepi16_epi32(x);
        __m512 y = _mm512_cvtepi32_ps(x_256);
        __m512 y_mul = _mm512_mul_ps(y, m_scale);
        _mm512_storeu_ps(outputData, y_mul);
	    outputData += 16;
        inputData += 32;
    }
    auto end = now();
    runtime_metrics.deQuant_time += (double(millisecondsDuration(end, start)));
    return true;
}
#endif


#else
#ifdef __AVX2__
bool quantizeToQuant8Signed(const float* inputData, int8_t* outputData, const Shape& outputShape
                            ) {
    uint32_t len = getNumberOfElements(outputShape.dimensions);
    uint32_t moves = len >> 4;
    uint32_t mod = len % 8;
    const __m512  m_scale =  _mm512_set1_ps(outputShape.scale);
    const __m512  m_offset =  _mm512_set1_ps(outputShape.offset);

    for ( int i = 0; i < moves; i++)
    {
        __m512 x = _mm512_loadu_ps(inputData);
        __m512 div_x = _mm512_div_ps(x, m_scale);
        __m512 add_x = _mm512_add_ps(div_x, m_offset);
        __m512i add_x_i = _mm512_cvt_roundps_epi32(add_x, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        __m128i x_int =  _mm512_cvtepi32_epi8(add_x_i);
         *((__m128i *) outputData) = x_int;
        inputData += 16;
        outputData += 16;
    }

    return true;
}

bool quantizeToQuant16(const float* inputData, uint16_t* outputData, const Shape& outputShape) {
    uint32_t len = getNumberOfElements(outputShape.dimensions);
    uint32_t moves = len >> 4;
    uint32_t mod = len % 8;
    const __m512 m_scale = _mm512_set1_ps(outputShape.scale);
    const __m512 m_offset = _mm512_set1_ps(outputShape.offset);

    for ( int i = 0; i < moves; i++)
    {
        __m512 x = _mm512_loadu_ps(inputData);
        __m512 div_x = _mm512_div_ps(x, m_scale);
        __m512 add_x = _mm512_add_ps(div_x, m_offset);
        __m512i x_int = _mm512_cvtps_epi32(add_x);
        __m256i x_int_16 = _mm512_cvtepi32_epi16(x_int);
         *((__m256i *) outputData) = x_int_16;
        inputData += 16;
        outputData += 16;
    }

    return true;
}
#elif defined(__SSE4_2__)
bool quantizeToQuant8Signed(const float* inputData, int8_t* outputData, const Shape& outputShape) {
    uint32_t len = getNumberOfElements(outputShape.dimensions);
    uint32_t moves = len >> 2;
    uint32_t mod = len % 8;
    const __m128 m_scale = _mm_set1_ps(outputShape.scale);
    const __m128 m_offset = _mm_set1_ps(outputShape.offset);

    for ( int i = 0; i < moves; i++)
    {
        __m128 x = _mm_loadu_ps(inputData);
        __m128 div_x = _mm_div_ps(x, m_scale);
        __m128 add_x = _mm_add_ps(div_x, m_offset);
        __m64 x_int = _mm_cvtps_pi8(add_x);
         *((__m64 *) outputData) = x_int;
        inputData += 4;
        outputData += 4;
    }

    return true;
}

bool quantizeToQuant16(const float* inputData, uint16_t* outputData, const Shape& outputShape) {
    uint32_t len = getNumberOfElements(outputShape.dimensions);
    uint32_t moves = len >> 2;
    uint32_t mod = len % 8;
    const __m128 m_scale = _mm_set1_ps(outputShape.scale);
    const __m128 m_offset = _mm_set1_ps(outputShape.offset);

    for ( int i = 0; i < moves; i++)
    {
        __m128 x = _mm_loadu_ps(inputData);
        __m128 div_x = _mm_div_ps(x, m_scale);
        __m128 add_x = _mm_add_ps(div_x, m_offset);
        __m64 x_int = _mm_cvtps_pi16(add_x);
         *((__m64 *) outputData) = x_int;
        inputData += 4;
        outputData += 4;
    }

    return true;
}
#endif

template<typename T>
bool deQuantize(const T* inputData, const uint32_t& len, const float scale,
                const int32_t zeroPoint, float* outputData) {
      int32_t value;
      for (int i = 0; i < len; ++i) {
        value = *(inputData + i);
        outputData[i] = static_cast<float>(scale * (value - zeroPoint));
      }

    return true;
}

#ifdef __AVX2__
template <>
bool deQuantize(const int8_t* inputData, const uint32_t& len, const float scale,
                const int32_t zeroPoint, float* outputData) {
    uint32_t moves = len >> 4;
    uint32_t mod = len % 8;
    const __m512 m_scale = _mm512_set1_ps(scale);
    const __m512 m_zp = _mm512_set1_ps(zeroPoint);
	for ( int i = 0; i < moves; i++)
    {
        __m128i x = _mm_load_si128((__m128i*)inputData);
        __m512i x_256 = _mm512_cvtepi8_epi32(x);
        __m512 y = _mm512_cvtepi32_ps(x_256);
        __m512 y_sub = _mm512_sub_ps(y, m_zp);
        __m512 y_mul = _mm512_mul_ps(y_sub, m_scale);
        _mm512_storeu_ps(outputData, y_mul);
	    outputData += 16;
        inputData += 16;
    }
    return true;
}

template <>
bool deQuantize(const uint16_t* inputData, const uint32_t& len, const float scale,
                const int32_t zeroPoint, float* outputData) {
    uint32_t moves = len >> 4;
    uint32_t mod = len % 8;
    const __m512 m_scale = _mm512_set1_ps(scale);
    auto start = now();
	for ( int i = 0; i < moves; i++)
    {
        __m256i x = _mm256_loadu_epi16(inputData);
        __m512i x_256 = _mm512_cvtepi16_epi32(x);
        __m512 y = _mm512_cvtepi32_ps(x_256);
        __m512 y_mul = _mm512_mul_ps(y, m_scale);
        _mm512_storeu_ps(outputData, y_mul);
	    outputData += 16;
        inputData += 32;
    }
    return true;
}
#elif defined(__SSE4_2__)
template <>
bool deQuantize(const uint16_t* inputData, const uint32_t& len, const float scale,
                const int32_t zeroPoint, float* outputData) {

    uint32_t moves = len >> 3;
    uint32_t mod = len % 8;
    const __m128i x0 = _mm_set1_epi16(0);
    const __m128 m_scale = _mm_set1_ps(scale);

	for ( int i = 0; i < moves; i++)
    {
        __m128i x = _mm_loadu_si128((__m128i*) inputData);
        __m128i xlo = _mm_unpacklo_epi16(x, _mm_cmplt_epi16 (x, x0));
        __m128i xhi = _mm_unpackhi_epi16(x, _mm_cmplt_epi16 (x, x0));
        __m128 ylo = _mm_cvtepi32_ps(xlo);
        __m128 yhi = _mm_cvtepi32_ps(xhi);
        __m128 valueslo = _mm_mul_ps(ylo, m_scale);
        __m128 valueshi = _mm_mul_ps(yhi, m_scale);

        _mm_storeu_ps(outputData, valueslo);
        _mm_storeu_ps(outputData + 4, valueshi);

        outputData += 8;
        inputData += 16;
    }
    return true;
}
#endif
#endif

Blob::Ptr GnaPreparedModel::getBlobFromMemoryPool(uint32_t index, const V1_3::Request& request, OperandType& opType) {
    RunTimeOperandInfo& operand = mOperands[mModelInputIndices[index]];
    const RequestArgument& arg = request.inputs[index];
    auto poolIndex = arg.location.poolIndex;
    nnAssert(poolIndex < mRuntimeRequestPoolInfos.size());
    auto& r = mRuntimeRequestPoolInfos[poolIndex];

    if (arg.dimensions.size() > 0) {
            // It's the responsibility of the cal
            // from.dimensions only modifies the dimensions that were
            // unspecified in the model.main.  That's the case in SampleDriver.cpp
            // with the call to validateRequest().
            operand.dimensions = arg.dimensions;
    }

    operand.buffer = r.buffer + arg.location.offset;
    operand.length = arg.location.length;
    opType = operand.type;

    return GetInOutOperandAsBlob(operand,
                                    const_cast<uint8_t*>(r.buffer + arg.location.offset),
                                    operand.length);
}

 RunTimeOperandInfo& GnaPreparedModel::getOperandFromMemoryPool(uint32_t index, const V1_3::Request& request) {
    RunTimeOperandInfo& operand = mOperands[mModelInputIndices[index]];
    const RequestArgument& arg = request.inputs[index];
    auto poolIndex = arg.location.poolIndex;
    nnAssert(poolIndex < mRuntimeRequestPoolInfos.size());
    auto& r = mRuntimeRequestPoolInfos[poolIndex];

    if (arg.dimensions.size() > 0) {
            // It's the responsibility of the caller to validate that
            // from.dimensions only modifies the dimensions that were
            // unspecified in the model.main.  That's the case in SampleDriver.cpp
            // with the call to validateRequest().
            operand.dimensions = arg.dimensions;
    }

    operand.buffer = r.buffer + arg.location.offset;
    operand.length = arg.location.length;

    return operand;
}

void GnaPreparedModel::executeGnaGraph() {
#ifdef PERF_COUNTERS
    auto gna_t0 = Time::now();
    gnaPluginPtr->Infer();
    auto gna_t1 = Time::now();
    fsec gna_fs = gna_t1 - gna_t0;
    ms d_gna = std::chrono::duration_cast<ms>(gna_fs);
    runtimeMetrics.infer_time.push_back(d_gna.count());
    gnaPluginPtr->inferTimeGNA.push_back(d_gna.count());
    gnaPluginPtr->noInferCall++;

    auto retPerfCounters = gnaPluginPtr->getInferRequest().GetPerformanceCounts();
    for (const auto &pair : retPerfCounters) {
        gnaPluginPtr->perfCounters[pair.first] = pair.second;
    }
    for (const auto &pair : gnaPluginPtr->perfCounters) {
        gnaPluginPtr->totalPerfCounters[pair.first].realTime_uSec += pair.second.realTime_uSec;
    }
#else
        gnaPluginPtr->Infer();
#endif
}

bool GnaPreparedModel::updateMemoryAfterGraphExecution(const V1_3::Request& request) {
    auto reqOutputs = request.outputs;
    for (auto i = 0; i < mModelOutputIndices.size(); i++) {
        auto index = mModelOutputIndices[i];
        RunTimeOperandInfo& operand = mOperands[index];
        const RequestArgument& arg = reqOutputs[i];
        auto poolIndex = arg.location.poolIndex;
        nnAssert(poolIndex < mRuntimeRequestPoolInfos.size());
        auto& r = mRuntimeRequestPoolInfos[poolIndex];

        void* destPtr = r.buffer + arg.location.offset;

        // Get the name of the layer from which we want to copy the data
        auto elementIdx = mOutputToLayerMap.find(index);
        if (elementIdx != mOutputToLayerMap.end()) {
            auto layerName = elementIdx->second.layerName;
            if (elementIdx->second.execDevice == DeviceType::GNA) {
                // TODO: Is this check needed???
                auto element = gnaPluginPtr->getOutputsInfo().find(layerName);
                if (element != gnaPluginPtr->getOutputsInfo().end()) {
                    Blob::Ptr outputBlob = gnaPluginPtr->getInferRequest().GetBlob(layerName);
                    float* srcPtr = outputBlob->buffer().as<float*>();
    #ifdef PERF_COUNTERS
                    if (operand.type == OperandType::TENSOR_QUANT16_SYMM) {
                        quantizeToQuant16(srcPtr, (uint16_t*)destPtr, operand.shape(), runtimeMetrics);
                    } else if (operand.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED) {
                        quantizeToQuant8Signed(srcPtr, (int8_t*)destPtr, operand.shape(), runtimeMetrics);
                    } else if (operand.type == OperandType::TENSOR_FLOAT32) {
                        std::memcpy((uint8_t*)destPtr, outputBlob->buffer().as<uint8_t*>(), outputBlob->byteSize());
                    }
    #else
                    if (operand.type == OperandType::TENSOR_QUANT16_SYMM) {
                        quantizeToQuant16(srcPtr, (uint16_t*)destPtr, operand.shape());
                    } else if (operand.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED) {
                        quantizeToQuant8Signed(srcPtr, (int8_t*)destPtr, operand.shape());
                    } else if (operand.type == OperandType::TENSOR_FLOAT32) {
                        std::memcpy((uint8_t*)destPtr, outputBlob->buffer().as<uint8_t*>(), outputBlob->byteSize());
                    }
    #endif
                } else {
                    VLOG(L1, "could not find layer:%s in index layer map", layerName.c_str());
                    return false;
                }
            } else {
                auto layerPtr = getCpuOpFromLayerName(layerName);
                // Check if the layername is present in the output map
                if (layerPtr) {
                    auto[srcPtrVoid, outputLen] = layerPtr->getOutputData();
    #ifdef PERF_COUNTERS
                    if (operand.type == OperandType::TENSOR_QUANT16_SYMM) {
                        quantizeToQuant16((float*)srcPtrVoid, (uint16_t*)destPtr, operand.shape(), runtimeMetrics);
                    } else if (operand.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED) {
                        quantizeToQuant8Signed((float*)srcPtrVoid, (int8_t*)destPtr, operand.shape(), runtimeMetrics);
                    } else if (operand.type == OperandType::TENSOR_FLOAT32) {
                        std::memcpy((uint8_t*)destPtr, (uint8_t*)srcPtrVoid, outputLen * sizeof(float));
                    }
    #else
                    if (operand.type == OperandType::TENSOR_QUANT16_SYMM) {
                        std::memcpy((uint8_t*)destPtr, (uint8_t*)srcPtrVoid, outputLen * sizeof(uint16_t));
                    } else if (operand.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED || operand.type == OperandType::TENSOR_QUANT8_ASYMM
                                || operand.type == OperandType::TENSOR_QUANT8_SYMM) {
                        std::memcpy((uint8_t*)destPtr, (uint8_t*)srcPtrVoid, outputLen * sizeof(uint8_t));
                    } else if (operand.type == OperandType::TENSOR_FLOAT32) {
                        std::memcpy((uint8_t*)destPtr, (uint8_t*)srcPtrVoid, outputLen * sizeof(float));
                    } else if (operand.type == OperandType::TENSOR_INT32) {
                        std::memcpy((uint8_t*)destPtr, (uint8_t*)srcPtrVoid, outputLen * sizeof(uint32_t));
                    }
    #endif
                } else {
                    VLOG(L1, "%s Failed to find the CPU layer by name: %s", __func__, layerName.c_str());
                    return false;
                }
            }
        } else {
            VLOG(L1, "could not find index:%d in output map", index);
            return false;
        }
    }

    return true;
}

bool GnaPreparedModel::updateMemoryAfterCPUGraphExecution(const V1_0_Request& request) {
    auto reqOutputs = request.outputs;
    for (auto i =0; i < mModelOutputIndices.size(); i++) {
        auto index = mModelOutputIndices[i];
        RunTimeOperandInfo& operand = mOperands[index];
        const RequestArgument& arg = reqOutputs[i];
        auto poolIndex = arg.location.poolIndex;
        nnAssert(poolIndex < mRuntimeRequestPoolInfos.size());
        auto& r = mRuntimeRequestPoolInfos[poolIndex];

        void* destPtr = r.buffer + arg.location.offset;
        std::vector<uint32_t> dimensions = arg.dimensions;
        // Get the name of the layer from which we want to copy the data
        auto elementIdx = mOutputToLayerMap.find(index);
        if (elementIdx != mOutputToLayerMap.end()) {
            auto layerName = elementIdx->second.layerName;
            auto layerPtr = getCpuOpFromLayerName(layerName);
            // Check if the layername is present in the output map
            if (layerPtr) {
                auto[srcPtrVoid, outputLen] = layerPtr->getOutputData();
#ifdef PERF_COUNTERS
                if (operand.type == OperandType::TENSOR_QUANT16_SYMM) {
                    quantizeToQuant16((float*)srcPtrVoid, (uint16_t*)destPtr, operand.shape(), runtimeMetrics);
                } else if (operand.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED) {
                    quantizeToQuant8Signed((float*)srcPtrVoid, (int8_t*)destPtr, operand.shape(), runtimeMetrics);
                } else if (operand.type == OperandType::TENSOR_FLOAT32) {
                    std::memcpy((uint8_t*)destPtr, (uint8_t*)srcPtrVoid, outputLen*4);
                }
#else
                if (operand.type == OperandType::TENSOR_QUANT16_SYMM) {
                    std::memcpy((uint8_t*)destPtr, (uint8_t*)srcPtrVoid, outputLen*sizeof(uint16_t));
                } else if (operand.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED) {
                    std::memcpy((uint8_t*)destPtr, (uint8_t*)srcPtrVoid, outputLen*sizeof(uint8_t));
                } else if (operand.type == OperandType::TENSOR_FLOAT32) {
                    std::memcpy((uint8_t*)destPtr, (uint8_t*)srcPtrVoid, outputLen*sizeof(float));
                }
#endif
            } else {
                VLOG(L1, "%s Failed to find the CPU layer by name: %s", __func__, layerName.c_str());
                return false;
            }
        } else {
            VLOG(L1, "could not find index:%d in output map", index);
            return false;
        }
    }

    return true;
}


bool GnaPreparedModel::updateMemoryAfterCPUGraphExecution(const V1_0_Request& request, uint32_t index) {
    uint32_t i =0;
    for (i =0; i < mModelOutputIndices.size(); i++) {
        if (mModelOutputIndices[i] == index)
            break;
    }

    auto reqOutputs = request.outputs;
    RunTimeOperandInfo& operand = mOperands[index];
    const RequestArgument& arg = reqOutputs[i];
    auto poolIndex = arg.location.poolIndex;
    nnAssert(poolIndex < mRuntimeRequestPoolInfos.size());
    auto& r = mRuntimeRequestPoolInfos[poolIndex];

    void* destPtr = r.buffer + arg.location.offset;

    // Get the name of the layer from which we want to copy the data
    auto elementIdx = mOutputToLayerMap.find(index);
    if (elementIdx != mOutputToLayerMap.end()) {
        auto layerName = elementIdx->second.layerName;
        auto layerPtr = getCpuOpFromLayerName(layerName);
        // Check if the layername is present in the output map
        if (layerPtr) {
            // Blob::Ptr outputBlob = gnaPluginPtr->getInferRequest().GetBlob(layerName);
            // float* srcPtr = outputBlob->buffer().as<float*>();
            auto[srcPtrVoid, outputLen] = layerPtr->getOutputData();
            // float* srcPtr = static_cast<float*>(srcPtrVoid);

            // VLOG(L1, "Copying output.. Output len:%d", outputLen);
            // for(auto i =0; i < 4; i++)
            //     VLOG(L1, "Copying output at index:%d is :%f", i, srcPtr[i]);

#ifdef PERF_COUNTERS
            if (operand.type == OperandType::TENSOR_QUANT16_SYMM) {
                quantizeToQuant16((float*)srcPtrVoid, (uint16_t*)destPtr, operand.shape(), runtimeMetrics);
            } else if (operand.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED) {
                quantizeToQuant8Signed((float*)srcPtrVoid, (int8_t*)destPtr, operand.shape(), runtimeMetrics);
            } else if (operand.type == OperandType::TENSOR_FLOAT32) {
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcPtrVoid, outputLen*4);
            }
#else
            if (operand.type == OperandType::TENSOR_QUANT16_SYMM) {
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcPtrVoid, outputLen*sizeof(uint16_t));
            } else if (operand.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED) {
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcPtrVoid, outputLen*sizeof(uint8_t));
            } else if (operand.type == OperandType::TENSOR_FLOAT32) {
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcPtrVoid, outputLen*sizeof(float));
            }
#endif
        } else {
            VLOG(L1, "%s Failed to find the CPU layer by name: %s", __func__, layerName.c_str());
            return false;
        }
    } else {
        VLOG(L1, "could not find index:%d in output map", index);
        return false;
    }

    return true;
}

void GnaPreparedModel::executeModel(const V1_3::Request& request) {
    // TODO: We need to change this to add more outputs
    // We are filling outputdata for only 1 output buffer for decoder
    //hidl_vec<OutputShape> outputShapes(request.outputs.size());
    hidl_vec<OutputShape> outputShapes(request.outputs.size());

    //TODO: Fix this code to make it generic
    std::vector<int8_t*> ptrsToDelete;

    for (auto index : mlayerInputIndices) {
        auto inputIndex = mModelInputIndices[index];
        auto iter = mInputPorts.find(inputIndex);
        if (iter != mInputPorts.end()) {
            std::string layerName = iter->second.layerName;
            RunTimeOperandInfo& operandInfo = getOperandFromMemoryPool(index, request);
            if (iter->second.execDevice == DeviceType::CPU) {
                auto OpPtr = getCpuOpFromLayerName(layerName);

                if (!OpPtr) {
                    VLOG(L1, "Failed to find the CPU layer by name: %s", layerName.c_str());
                    nnAssert(false);
                }

                auto length = getNumberOfElements(operandInfo.shape().dimensions);
                OpPtr->setInputData(inputIndex, operandInfo.buffer,
                                            length);
            } else {
                OperandType opType;
                auto srcBlob = getBlobFromMemoryPool(index, request, opType);

                if (iter->second.memoryLayer) {
                    gnaPluginPtr->setMemoryState(layerName, srcBlob);
                } else {
                    VLOG(L1, "Setting state\n");
                    // Make sure the layername is present in inputinfo from the layer
                    auto iter2 = std::find_if(gnaPluginPtr->inputInfo.begin(),
                                gnaPluginPtr->inputInfo.end(),
                                [layerName](const std::pair<std::string, InputInfo::Ptr>& elem){
                                    return (elem.first == layerName);
                                });
                    if (iter2 == gnaPluginPtr->inputInfo.end()) {
                        nnAssert("Input index does not have a input layer");
                    }
                    auto destBlob = gnaPluginPtr->getInferRequest().GetBlob(layerName);
                    RunTimeOperandInfo& current_op = mOperands[inputIndex];
                    uint8_t* dest = destBlob->buffer().as<uint8_t*>();
                    uint8_t* src = srcBlob->buffer().as<uint8_t*>();
                    std::memcpy(dest, src, srcBlob->byteSize());
                    srcBlob->deallocate();
                }
            }
        } else {
            ALOGE("Index:%d not found in input layers map", inputIndex);
        }
    }
    VLOG(L1, "Run");

    auto isInputIndexIntermediateOutput = [&](uint32_t index) -> bool {
        if ((mIntermediateLayerMap.find(index) != mIntermediateLayerMap.end())) {
            auto elem = mIntermediateLayerMap.at(index);

            if ((elem.inDevice == DeviceType::None) || (elem.outDevice == DeviceType::None)) {
                return false;
            }

            if ((elem.inDevice == elem.outDevice)) {
                return false;
            }
            return true;
        }
        return false;
    };

    for (auto opcontainer: mNwManager) {
        if (opcontainer->isCpuGraph()) {
            std::vector<uint32_t> inputsIndex = opcontainer->getInputIndices();
            for(auto inputInd : inputsIndex) {
                if (isInputIndexIntermediateOutput(inputInd)) {
                    auto elementIdx = mIntermediateLayerMap.find(inputInd);
                    if (elementIdx != mIntermediateLayerMap.end()) {
                        auto gnaLayerName = elementIdx->second.outLayerName;
                        auto cpuLayerName = elementIdx->second.inLayerName;
                        RunTimeOperandInfo& op = mOperands[inputInd];
                        // Check if the layername is present in the output map
                        auto element = gnaPluginPtr->getOutputsInfo().find(gnaLayerName);
                        if (element != gnaPluginPtr->getOutputsInfo().end()) {
                            Blob::Ptr srcBlob = gnaPluginPtr->getInferRequest().GetBlob(gnaLayerName);

                            if (op.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED) {
                                BaseOp* currOp = opcontainer->getCpuOpFromLayerName(cpuLayerName);
                                if (!dummyOpMap[inputInd]) {
                                    int8_t* destPtr = new int8_t[srcBlob->size()];
                                    ptrsToDelete.emplace_back(destPtr);
                                    float * opGetBlob = srcBlob->buffer().as<float*>();
                                    #ifdef PERF_COUNTERS
                                        quantizeToQuant8Signed(srcBlob->buffer().as<float*>(),
                                                                (int8_t*)destPtr, op.shape(), runtimeMetrics);
                                    #else
                                        quantizeToQuant8Signed(srcBlob->buffer().as<float*>(),
                                                                (int8_t*)destPtr, op.shape());
                                    #endif
                                    currOp->setInputData(inputInd, destPtr, srcBlob->size());
                                } else {
                                    currOp->setInputData(inputInd, srcBlob->buffer().as<void*>(), srcBlob->size());
                                }
                            } else {
                                VLOG(L1, "op type for copying to CPU is different from TENSOR_QUANT8_ASYMM_SIGNED !!!!");
								nnAssert(false);
                            }
                        } else {
                            VLOG(L1, "Unable to find GNA Layer for CPU graph!!");
                        }
                    }
                }
            }
            opcontainer->run();
        } else {
            std::vector<uint32_t> inputsIndex = opcontainer->getInputIndices();
            for(auto inputInd : inputsIndex) {
                // TODO: We need to also search for SUBGRAPH_OUTPUT as well..
                if (isInputIndexIntermediateOutput(inputInd)) {
                    auto elementIdx = mIntermediateLayerMap.find(inputInd);
                    if (elementIdx != mIntermediateLayerMap.end()) {
                        auto cpuLayerName = elementIdx->second.outLayerName;
                        auto gnaLayerName = elementIdx->second.inLayerName;
                        auto cpuLayerPtr = getCpuOpFromLayerName(cpuLayerName);
                        if (!cpuLayerPtr) {
                            VLOG(L1, "Could not find the layer name in cpu layers %s", cpuLayerName.c_str());
                            nnAssert(false);
                        }
                        auto[srcMemoryPtr, srcLen] = cpuLayerPtr->getOutputData();

                        // Make sure the layername is present in inputinfo from the layer
                        auto iter2 = std::find_if(gnaPluginPtr->inputInfo.begin(),
                                    gnaPluginPtr->inputInfo.end(),
                                    [gnaLayerName](const std::pair<std::string, InputInfo::Ptr>& elem){
                                        return (elem.first == gnaLayerName);
                                    });
                        if (iter2 == gnaPluginPtr->inputInfo.end()) {
                            VLOG(L1, "Could not find the layername:%s in GNA layer inputs", cpuLayerName.c_str());
                            nnAssert(false);
                        }

                        auto destBlob = gnaPluginPtr->getInferRequest().GetBlob(gnaLayerName);
                        RunTimeOperandInfo& op = mOperands[inputInd];
                        if (op.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED) {
                            if (isRnnT() && !isJointNw) {
                                float* srcPtr = static_cast<float*>(srcMemoryPtr);
                                std::memcpy(destBlob->buffer().as<float*>(), srcPtr, srcLen * 4);
                            }
                            else {
                                int8_t* srcPtr = static_cast<int8_t*>(srcMemoryPtr);
                                deQuantize<int8_t>(srcPtr, srcLen, op.scale, op.zeroPoint, destBlob->buffer().as<float*>());
                            }
                        }
                    } else {
                        VLOG(L1, "Index  %d is not in output index map", index);
                    }
                }
            }

            executeGnaGraph();
            // if (!updateMemoryAfterGNAGraphExecution(request)) {
            //     VLOG(L1, "Failed to update memory after GNA graph execution!!!!!");
            //     nnAssert(false);
            // }
        }
    }

    updateMemoryAfterGraphExecution(request);

    for (auto opcontainer: mNwManager) {
        opcontainer->cleanup();
    }

    for (auto ptr: ptrsToDelete) {
        delete[] ptr;
    }

    for (auto runtimeInfo : mRuntimeRequestPoolInfos) {
        runtimeInfo.update();
    }

    for (auto runtimeInfo : mRuntimeRequestPoolInfos) {
        runtimeInfo.unmap_mem();
    }

    return;
}

std::tuple<ErrorStatus, hidl_vec<V1_2::OutputShape>, Timing>
 GnaPreparedModel::syncExecute(const V1_3::Request& request, MeasureTiming measure, time_point driverStart) {

#ifdef PERF_COUNTERS
    runtimeMetrics.infer_calls++;
#endif
    time_point driverEnd, deviceStart, deviceEnd;
    if (measure == MeasureTiming::YES) deviceStart = now();

    if (!setRunTimePoolInfosFromHidlMemories(&mRuntimeRequestPoolInfos, request.pools)) {
        return {ErrorStatus::GENERAL_FAILURE, {}, kNoTiming};
    }

    executeModel(request);

    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(deviceEnd, deviceStart))};
        VLOG(L1, "Driver::asyncExecute timing = %s", toString(timing).c_str());
    } else {
        VLOG(L1, "MeasureTiming - No. Returning with no error");
    }

    return {ErrorStatus::NONE, {}, kNoTiming};
}

void GnaPreparedModel::asyncExecute(const V1_3::Request& request, MeasureTiming measure, time_point driverStart,
                          const sp<V1_3::IExecutionCallback>& cb) {
#ifdef PERF_COUNTERS
    runtimeMetrics.infer_calls++;
#endif
    time_point driverEnd, deviceStart, deviceEnd;
    if (measure == MeasureTiming::YES) deviceStart = now();

    if (!setRunTimePoolInfosFromHidlMemories(&mRuntimeRequestPoolInfos, request.pools)) {
        cb->notify(V1_0::ErrorStatus::GENERAL_FAILURE);
        return;
    }

    executeModel(request);

    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(deviceEnd, deviceStart))};
        VLOG(L1, "Driver::asyncExecute timing = %s", toString(timing).c_str());
    } else {
        VLOG(L1, "MeasureTiming - No. Returning with no error");
    }

    cb->notify(V1_0::ErrorStatus::NONE);
}

Return<V1_0::ErrorStatus> GnaPreparedModel::executeBase(const V1_0::Request& request, MeasureTiming measure,
                                               const sp<V1_0::IExecutionCallback>& callback) {
    return V1_0::ErrorStatus::NONE;
}

Return<V1_3::ErrorStatus> GnaPreparedModel::executeBase_V1_3(const V1_3::Request& request, MeasureTiming measure,
                                               const sp<V1_3::IExecutionCallback>& callback) {
    VLOG(L1, "executebase 1_3");

    time_point driverStart;
    if (measure == MeasureTiming::YES) driverStart = now();

    if (callback.get() == nullptr) {
        ALOGE("invalid callback passed to execute");
        return V1_3::ErrorStatus::INVALID_ARGUMENT;
    }

//TODO: Add back ValidateRequest
#if 0
    if (!validateRequest(request, convertToV1_0(mModel))) {
        notify(callback, V1_0_ErrorStatus::INVALID_ARGUMENT, {}, kNoTiming);
        return V1_0_ErrorStatus::INVALID_ARGUMENT;
    }
#endif
    // This thread is intentionally detached because the driver service
    // is expected to live forever.
    // std::thread([this, request, measure, driverStart, callback] {
    //     asyncExecute(request, measure, driverStart, callback);
    // }).detach();
    asyncExecute(request, measure, driverStart, callback);

    return V1_3::ErrorStatus::NONE;
}

void GnaPreparedModel::setDims(const uint32_t idx, const std::vector<unsigned long> dims) {
    auto &op = mModel.main.operands[idx];
    int i = 0;

    for (auto &item : op.dimensions) {
        item = dims[i];
        i++;
    }
}

bool GnaPreparedModel::operationAdd(const Operation& operation) {
    VLOG(L1, "OperationType::ADD");

    auto getV1_3_OperandLifeTime = [&](uint32_t idx) {
        const auto op = mModel.main.operands[idx];
        return (int)op.lifetime;
    };

    auto getIRBlobFromOperand = [&](uint32_t idx, uint32_t offset) {
        const auto op = mModel.main.operands[idx];

        auto blob = GetConstOperandAsTensor(idx, offset);
        if (op.lifetime == V1_3_OperandLifeTime::SUBGRAPH_INPUT)
        {
            mOpIndex2BlobMap[idx] = blob;
        }

        return blob;
    };

    IRBuilder::BuilderADDLayer::AddParams params;
    params.input1.data = getIRBlobFromOperand(operation.inputs[0], 0);
    params.input1.lifeTime = getV1_3_OperandLifeTime(operation.inputs[0]);

    params.input2.data = getIRBlobFromOperand(operation.inputs[1], 1);
    params.input2.lifeTime = getV1_3_OperandLifeTime(operation.inputs[1]);

    auto input1Dims = params.input1.data->getTensorDesc().getDims();
    auto input2Dims = params.input2.data->getTensorDesc().getDims();
    IRBlob::Ptr input = nullptr;

    for (auto i = 0; i < input2Dims.size(); i++) {
        VLOG(L1, "input2Dims dims[%d] = %d ", i, input2Dims[i]);
        VLOG(L1, "input1Dims dims[%d] = %d ", i, input1Dims[i]);
    }
    nnAssert(input2Dims.size() == input1Dims.size());

    if (mBuilderModel == nullptr) {
        VLOG(L1, "mBuilder = nullptr !!!");
    }

    std::vector<std::string> inLayers;
    std::string addLayerName = mBuilderModel->createAdd(params, nullptr, inLayers);

    // Create an OUTPUT layer for Add.
    static int count = 0;
    std::string outputLayerName = "output-ADD";
    outputLayerName += std::to_string(count++);

	auto addLayerID = mBuilderModel->getBuilderNetwork()->mConnections.back();
	mBuilderModel->getBuilderNetwork()->getBuilder()->addLayer({addLayerID}, InferenceEngine::Builder::OutputLayer(outputLayerName));

    // TODO: Fix this code with CTS tests
    // Check CONST_REFERENCE, CONST_VALUE
    if (inLayers.size() != 0) {
        for (int layer_num = 0; layer_num < inLayers.size(); layer_num++) {
            if (getV1_3_OperandLifeTime(operation.inputs[layer_num]) == (int)V1_3_OperandLifeTime::SUBGRAPH_INPUT) {
                mInputPorts.emplace(std::make_pair(operation.inputs[layer_num], LayerInfo(inLayers[layer_num], false)));
            } else if (getV1_3_OperandLifeTime(operation.inputs[layer_num]) == (int)V1_3_OperandLifeTime::TEMPORARY_VARIABLE) {
                VLOG(L1, "add is temp input , setting the intermediate Layer %d\n", operation.inputs[layer_num]);

                if (mIntermediateLayerMap.find(operation.inputs[layer_num]) != mIntermediateLayerMap.end()) {
                    VLOG(L1, "add is temp input , setting the intermediate Layer element found %d\n", operation.inputs[layer_num]);

                    auto& halLayer = mIntermediateLayerMap.at(operation.inputs[layer_num]);
                    halLayer.setInputNode(inLayers[layer_num], DeviceType::GNA);
                } else {
                    VLOG(L1, "add is temp input , setting the intermediate Layer %d\n", operation.inputs[layer_num]);

                    mIntermediateLayerMap.emplace(std::make_pair(operation.inputs[layer_num],
                                                                HalLayerInfo(inLayers[layer_num], DeviceType::GNA,
                                                                            "", DeviceType::None, false)));
                }
            }
        }
    }

    if (getV1_3_OperandLifeTime(operation.outputs[0]) == (int)V1_3_OperandLifeTime::SUBGRAPH_OUTPUT) {
        mOutputToLayerMap.emplace(std::make_pair(operation.outputs[0], LayerInfo(addLayerName, false, DeviceType::GNA)));
    } else if (static_cast<int>(getV1_3_OperandLifeTime(operation.outputs[0])) == static_cast<int>(OperandLifeTime::TEMPORARY_VARIABLE)) {
        if (mIntermediateLayerMap.find(operation.outputs[0]) != mIntermediateLayerMap.end()) {
            auto& halLayer = mIntermediateLayerMap.at(operation.outputs[0]);
            halLayer.setOutputNode(addLayerName, DeviceType::GNA);
        } else {
            mIntermediateLayerMap.emplace(std::make_pair(operation.outputs[0], HalLayerInfo("", DeviceType::None,
                                                                            addLayerName, DeviceType::GNA, false)));
        }
    }

    if (gnaPluginPtr) {
        gnaPluginPtr->setInputIndices({operation.inputs[0], operation.inputs[1]});
        gnaPluginPtr->setOutputIndices({operation.outputs[0]});
    }


    return true;
}

bool GnaPreparedModel::operationTANH(const Operation& operation) {
    VLOG(L1, "OperationType::TANH");

    auto getV1_3_OperandLifeTime = [&](uint32_t idx) {
        const auto op = mModel.main.operands[idx];
        return (int)op.lifetime;
    };

    auto getIRBlobFromOperand = [&](uint32_t idx, uint32_t offset) {
        const auto op = mModel.main.operands[idx];

        auto blob = GetConstOperandAsTensor(idx, offset);
        if (op.lifetime == V1_3_OperandLifeTime::SUBGRAPH_INPUT)
        {
            mOpIndex2BlobMap[idx] = blob;
        }

        return blob;
    };

    IRBuilder::BuilderTANHLayer::TanhParams params;
    params.input.data = getIRBlobFromOperand(operation.inputs[0], 0);
    params.input.lifeTime = getV1_3_OperandLifeTime(operation.inputs[0]);

    auto inputDims = params.input.data->getTensorDesc().getDims();
    IRBlob::Ptr input = nullptr;

    for (auto i = 0; i < inputDims.size(); i++) {
        VLOG(L1, "inputDims dims[%d] = %d ", i, inputDims[i]);
    }


    if (mBuilderModel == nullptr) {
        VLOG(L1, "mBuilder = nullptr !!!");
        // ASSERT
    }

    std::vector<std::string> inLayers;
    std::string TanhLayerName = mBuilderModel->createTanh(params, nullptr, inLayers);

    static int count = 0;
    std::string outputLayerName = "output-Tanh";
    outputLayerName += std::to_string(count++);

	auto TanhLayerID = mBuilderModel->getBuilderNetwork()->mConnections.back();
	mBuilderModel->getBuilderNetwork()->getBuilder()->addLayer({TanhLayerID}, InferenceEngine::Builder::OutputLayer(outputLayerName));

    // TODO: Fix this code with CTS tests
    if (inLayers.size() != 0) {
        for (int layer_num = 0; layer_num < inLayers.size(); layer_num++) {
            if (getV1_3_OperandLifeTime(operation.inputs[layer_num]) == (int)V1_3_OperandLifeTime::SUBGRAPH_INPUT) {
                mInputPorts.emplace(std::make_pair(operation.inputs[layer_num], LayerInfo(inLayers[layer_num], false)));
            } else if (getV1_3_OperandLifeTime(operation.inputs[layer_num]) == (int)V1_3_OperandLifeTime::TEMPORARY_VARIABLE) {
                VLOG(L1, "Tanh is temp input , setting the intermediate Layer %d\n", operation.inputs[layer_num]);

                if (mIntermediateLayerMap.find(operation.inputs[layer_num]) != mIntermediateLayerMap.end()) {
                    VLOG(L1, "Tanh is temp input , setting the intermediate Layer element found %d\n", operation.inputs[layer_num]);

                    auto& halLayer = mIntermediateLayerMap.at(operation.inputs[layer_num]);
                    halLayer.setInputNode(inLayers[layer_num], DeviceType::GNA);
                } else {
                    VLOG(L1, "Tanh is temp input , setting the intermediate Layer %d\n", operation.inputs[layer_num]);

                    mIntermediateLayerMap.emplace(std::make_pair(operation.inputs[layer_num],
                                                                HalLayerInfo(inLayers[layer_num], DeviceType::GNA,
                                                                            "", DeviceType::None, false)));
                }
            }
        }
    }

    if (getV1_3_OperandLifeTime(operation.outputs[0]) == (int)V1_3_OperandLifeTime::SUBGRAPH_OUTPUT) {
        mOutputToLayerMap.emplace(std::make_pair(operation.outputs[0], LayerInfo(TanhLayerName, false, DeviceType::GNA)));
    } else if (static_cast<int>(getV1_3_OperandLifeTime(operation.outputs[0])) == static_cast<int>(OperandLifeTime::TEMPORARY_VARIABLE)) {
        if (mIntermediateLayerMap.find(operation.outputs[0]) != mIntermediateLayerMap.end()) {
            auto& halLayer = mIntermediateLayerMap.at(operation.outputs[0]);
            halLayer.setOutputNode(TanhLayerName, DeviceType::GNA);
        } else {
            mIntermediateLayerMap.emplace(std::make_pair(operation.outputs[0], HalLayerInfo("", DeviceType::None,
                                                                            TanhLayerName, DeviceType::GNA, false)));
        }
    }

    if (gnaPluginPtr) {
        gnaPluginPtr->setInputIndices({operation.inputs[0]});
        gnaPluginPtr->setOutputIndices({operation.outputs[0]});
    }


    return true;
}

bool GnaPreparedModel::operationFullyConnected(const Operation& operation) {
    VLOG(L1, "OperationType::FULLY_CONNECTED");

    auto getV1_3_OperandLifeTime = [&](uint32_t idx) {
        const auto op = mModel.main.operands[idx];
        return (int)op.lifetime;
    };

    auto getIRBlobFromOperand = [&](uint32_t idx, uint32_t offset) {
        const auto op = mModel.main.operands[idx];
        auto blob = GetConstOperandAsTensor(idx, offset);
        if (op.lifetime == V1_3_OperandLifeTime::SUBGRAPH_INPUT)
        {
            mOpIndex2BlobMap[idx] = blob;
        }

        return blob;
    };

    auto validateOperand = [&](uint32_t idx, uint32_t offset) {
        const auto op = mModel.main.operands[idx];
        auto len_out = op.location.length;
        if (len_out == 0) {
            return false;
        }
        return true;
    };

    IRBuilder::BuilderFCLayer::FCParams params;
    params.input.lifeTime = getV1_3_OperandLifeTime(operation.inputs[0]);
    params.input.data = getIRBlobFromOperand(operation.inputs[0], 0);

    params.weights.data = getIRBlobFromOperand(operation.inputs[1], 1);
    params.weights.lifeTime = getV1_3_OperandLifeTime(operation.inputs[1]);

    params.bias.data = getIRBlobFromOperand(operation.inputs[2], 2);
    params.bias.lifeTime = getV1_3_OperandLifeTime(operation.inputs[2]);

    uint32_t len;
    params.fuse_parameter = *((uint32_t*)(GetOperandMemory(mModel, operation.inputs[3], len)));
    auto inputDims = params.input.data->getTensorDesc().getDims();
    IRBlob::Ptr input = nullptr;

    auto weightsDims = params.weights.data->getTensorDesc().getDims();
    for (auto i = 0; i < weightsDims.size(); i++)
        VLOG(L1, "weights dims[%d] = %d ", i, weightsDims[i]);

    auto biasDims = params.bias.data->getTensorDesc().getDims();
    setDims(operation.outputs[0], inputDims);

    // input is [batch_size, input_size], weights is [num_unit, input_size]
    // nnAssert(inputDims[1] == weightsDims[1]);
    nnAssert(weightsDims.size() == 2);
    uint32_t numInputElements = sizeOfTensor(inputDims);
    uint32_t num_units = weightsDims[0];
    uint32_t input_size = weightsDims[1];
    uint32_t batch_size = numInputElements / input_size;
    nnAssert(biasDims[0] == num_units);
    nnAssert(input_size * batch_size == numInputElements);

    if (mBuilderModel == nullptr) {
        VLOG(L1, "mBuilder = nullptr !!!");
        // ASSERT
    }

    auto getLayerName = [&](std::string layerName) -> std::string
    {
        std::string strName(layerName);
        strName = strName + "_" + std::to_string(mBuilderModel->layer_name_count++);
        return strName;
    };


    std::vector<std::string> inLayers;
    std::string fcLayerName = mBuilderModel->createFC(params, nullptr, inLayers);
    if (fcLayerName.empty()) {
        return false;
    }
    // Create an OUTPUT layer for FC.
    static int count = 0;
    std::string outputLayerName = "output-FC";
    outputLayerName += std::to_string(count++);

	auto fcLayerId = mBuilderModel->getBuilderNetwork()->mConnections.back();
    idx_t reluId =fcLayerId;

    if (params.fuse_parameter  == (int32_t)FusedActivationFunc::RELU) {
        fcLayerName = getLayerName("fused_relu");
        reluId = mBuilderModel->getBuilderNetwork()->getBuilder()->addLayer({fcLayerId}, InferenceEngine::Builder::ReLULayer(fcLayerName) \
                             .setPort(Port({inputDims[0], weightsDims[1] * weightsDims[0]/inputDims[1]})));
        mBuilderModel->getBuilderNetwork()->mConnections.push_back(reluId);
    }

	mBuilderModel->getBuilderNetwork()->getBuilder()->addLayer({reluId}, InferenceEngine::Builder::OutputLayer(outputLayerName));

    // TODO: Fix this code with CTS tests
    if (inLayers.size() != 0) {
        if (getV1_3_OperandLifeTime(operation.inputs[0]) == (int)V1_3_OperandLifeTime::SUBGRAPH_INPUT) {
            mInputPorts.emplace(std::make_pair(operation.inputs[0], LayerInfo(inLayers[0], false)));
        } else if (getV1_3_OperandLifeTime(operation.inputs[0]) == (int)V1_3_OperandLifeTime::TEMPORARY_VARIABLE) {
            if (mIntermediateLayerMap.find(operation.inputs[0]) != mIntermediateLayerMap.end()) {
                auto& halLayer = mIntermediateLayerMap.at(operation.inputs[0]);
                halLayer.setInputNode(inLayers[0], DeviceType::GNA);
            } else {
                mIntermediateLayerMap.emplace(std::make_pair(operation.inputs[0],
                                                            HalLayerInfo(inLayers[0], DeviceType::GNA,
                                                                         "", DeviceType::None, false)));
            }
        }
    }

    if (getV1_3_OperandLifeTime(operation.outputs[0]) == (int)V1_3_OperandLifeTime::SUBGRAPH_OUTPUT) {
        mOutputToLayerMap.emplace(std::make_pair(operation.outputs[0], LayerInfo(fcLayerName, false, DeviceType::GNA)));
    } else if (static_cast<int>(getV1_3_OperandLifeTime(operation.outputs[0])) == static_cast<int>(OperandLifeTime::TEMPORARY_VARIABLE)) {
        if (mIntermediateLayerMap.find(operation.outputs[0]) != mIntermediateLayerMap.end()) {
            auto& halLayer = mIntermediateLayerMap.at(operation.outputs[0]);
            halLayer.setOutputNode(fcLayerName, DeviceType::GNA);
        } else {
            mIntermediateLayerMap.emplace(std::make_pair(operation.outputs[0], HalLayerInfo("", DeviceType::None,
                                                                            fcLayerName, DeviceType::GNA, false)));
        }
    }

    VLOG(L1, "----------------------------------------------");
    VLOGDIMS(L1, inputDims, "inputs dims");
    //VLOGDIMS(L1, newInputDims, "newInput dims");
    VLOGDIMS(L1, weightsDims, "weights dims");
    VLOG(L1, "----------------------------------------------");

    return true;
}

/*
    Performs a single time step in a Long Short-Term Memory (LSTM) layer.

    The LSTM operation is described by the following equations.

    \begin{eqnarray*} i_t =& \sigma(W_{xi}x_t+W_{hi}h_{t-1}+W_{ci}C_{t-1}+b_i) &
    \\ f_t =& \sigma(W_{xf}x_t+W_{hf}h_{t-1}+W_{cf}C_{t-1}+b_f) &
    \\ C_t =& clip(f_t \odot C_{t-1} + i_t \odot g(W_{xc}x_t+W_{hc}h_{t-1}+b_c),\ t_{cell}) &
    \\ o_t =& \sigma(W_{xo}x_t+W_{ho}h_{t-1}+W_{co}C_t+b_o) & \\ & &
    \\ & clip(W_{proj}(o_t \odot g(C_t))+b_{proj},\ t_{proj})
    & if\ there\ is\ a\ projection; \\ h_t =& & \\ & o_t \odot g(C_t) & otherwise.

Inputs:

[23]{30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52}
    0: The input ( $x_t$). A 2-D tensor of shape [batch_size, input_size], where “batch_size” corresponds to the batching dimension,
    and “input_size” is the size of the input.
    1: The input-to-input weights ( $W_{xi}$). Optional. A 2-D tensor of shape [num_units, input_size], where “num_units” corresponds to the number of cell units.
    2: The input-to-forget weights ( $W_{xf}$). A 2-D tensor of shape [num_units, input_size].
    3: The input-to-cell weights ( $W_{xc}$). A 2-D tensor of shape [num_units, input_size].
    4: The input-to-output weights ( $W_{xo}$). A 2-D tensor of shape [num_units, input_size].
    5: The recurrent-to-input weights ( $W_{hi}$). Optional. A 2-D tensor of shape [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e., “num_units”), or the second dimension of the “projection_weights”, if defined.
    6: The recurrent-to-forget weights ( $W_{hf}$). A 2-D tensor of shape [num_units, output_size].
    7: The recurrent-to-cell weights ( $W_{hc}$). A 2-D tensor of shape [num_units, output_size].
    8: The recurrent-to-output weights ( $W_{ho}$). A 2-D tensor of shape [num_units, output_size].
    9: The cell-to-input weights ( $W_{ci}$). Optional. A 1-D tensor of shape [num_units].
    10:The cell-to-forget weights ( $W_{cf}$). Optional. A 1-D tensor of shape [num_units].
    11:The cell-to-output weights ( $W_{co}$). Optional. A 1-D tensor of shape [num_units].
    12:The input gate bias ( $b_i$). Optional. A 1-D tensor of shape [num_units].
    13:The forget gate bias ( $b_f$). A 1-D tensor of shape [num_units].
    14:The cell bias ( $b_c$). A 1-D tensor of shape [num_units].
    15:The output gate bias ( $b_o$). A 1-D tensor of shape [num_units].
    16:The projection weights ( $W_{proj}$). Optional. A 2-D tensor of shape [output_size, num_units].
    17:The projection bias ( $b_{proj}$). Optional. A 1-D tensor of shape [output_size].
    18:The output state (in) ( $h_{t-1}$). A 2-D tensor of shape [batch_size, output_size].
    19:The cell state (in) ( $C_{t-1}$). A 2-D tensor of shape [batch_size, num_units].
    20:The activation function ( $g$). A value indicating the activation function:
        0: None;
        1: Relu;
        3: Relu6;
        4: Tanh;
        6: Sigmoid.
    21:The clipping threshold ( $t_{cell}$) for the cell state, such that values are bound within [-cell_clip, cell_clip]. If set to 0.0 then clipping is disabled. Until API level 29 this scalar must be of type ANEURALNETWORKS_FLOAT32. Since API level 29, if all the input tensors have type ANEURALNETWORKS_TENSOR_FLOAT32, this scalar must be of the type ANEURALNETWORKS_FLOAT32, otherwise if all the input tensors have the type ANEURALNETWORKS_TENSOR_FLOAT16, this scalar must be of type ANEURALNETWORKS_FLOAT16.
    22:The clipping threshold ( $t_{proj}$) for the output from the projection layer, such that values are bound within [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled. Until API level 29 this scalar must be of type ANEURALNETWORKS_FLOAT32. Since API level 29, if all the input tensors have type ANEURALNETWORKS_TENSOR_FLOAT32, this scalar must be of the type ANEURALNETWORKS_FLOAT32, otherwise if all the input tensors have the type ANEURALNETWORKS_TENSOR_FLOAT16, this scalar must be of type ANEURALNETWORKS_FLOAT16. Since API level 29 there are additional inputs to this op:
    23:The input layer normalization weights. A 1-D tensor of shape [num_units]. Used to rescale normalized inputs to activation at input gate.
    24:The forget layer normalization weights. A 1-D tensor of shape [num_units]. Used to rescale normalized inputs to activation at forget gate.
    25:The cell layer normalization weights. A 1-D tensor of shape [num_units]. Used to rescale normalized inputs to activation at cell gate.
    26:The output layer normalization weights. A 1-D tensor of shape [num_units]. Used to rescale normalized inputs to activation at output gate.

Outputs:

    0: The scratch buffer. A 2-D tensor of shape [batch_size, num_units * 3] with CIFG, or [batch_size, num_units * 4] without CIFG.
    1: The output state (out) ( $h_t$). A 2-D tensor of shape [batch_size, output_size].
    2: The cell state (out) ( $C_t$). A 2-D tensor of shape [batch_size, num_units].
    3: The output ( $o_t$). A 2-D tensor of shape [batch_size, output_size]. This is effectively the same as the current “output state (out)” value.

    Available since API level 27.
*/
bool GnaPreparedModel::operationLSTM(const Operation& operation)
{
    IRBuilder::LstmLayer::LstmParams  params;

    auto getV1_3_OperandLifeTime = [&](uint32_t idx) {
        const auto op = mModel.main.operands[idx];
        return (int)op.lifetime;
    };

    auto getIRBlobFromOperand = [&](uint32_t idx, uint32_t offset) {
        const auto op = mModel.main.operands[idx];

        auto blob = GetConstOperandAsTensor(idx, offset);
        if (op.lifetime == V1_3_OperandLifeTime::SUBGRAPH_INPUT)
        {
            mOpIndex2BlobMap[idx] = blob;
            VLOG(L1, "blob idx=%d (model_input) ptr=%p", idx, blob.get());
        }

        return blob;
    };

    IRBuilder::LstmLayer::LstmCellDescription lstmDesc;
	lstmDesc.clippingThresholdCellState     = 0;
    lstmDesc.clippingThresholdProjState     = 0;
    lstmDesc.cifgEnabled                    = false;
    lstmDesc.projectionLayerEnabled         = false;
    lstmDesc.peepholeEnabled                = false;

    lstmDesc.clippingThresholdCellState     = PARAM_FP(21);
    lstmDesc.clippingThresholdProjState     = PARAM_FP(22);

    std::string lstmDescription;
    if (isOperandDataNull(operation.inputs[1]) ||
        isOperandDataNull(operation.inputs[5]) ||
        isOperandDataNull(operation.inputs[12])) {
        lstmDescription.append("Cifg");
        lstmDesc.cifgEnabled = true;

        VLOG(L1, "CIFG enabled !!!!!!!!");

        if (isOperandDataNull(operation.inputs[20])) {
            VLOG(L1, "Input 20 is null!!!!");
        }

        if (isOperandDataNull(operation.inputs[26])) {
            VLOG(L1, "Input 26 is null!!!!");
        }
    } else {
        lstmDescription.append("noCifg");
    }

    if (!isOperandDataNull(operation.inputs[16]))
    {
        lstmDescription.append("Projection");
        lstmDesc.projectionLayerEnabled = true;
    } else {
        lstmDescription.append("noProjection");
    }

    if (!isOperandDataNull(operation.inputs[9]) ||
        !isOperandDataNull(operation.inputs[10]) ||
        !isOperandDataNull(operation.inputs[11]))
    {
        lstmDescription.append("Peephole");
        lstmDesc.peepholeEnabled = true;
    } else {
        lstmDescription.append("noPeephole");
    }

    params.useLayerNorm = false;
    if (operation.inputs.size() == 27) {
        params.useLayerNorm = true;
        params.useBatchedLayerNorm = true;
    }

    VLOG(L1, "Lstm cell description %s", lstmDescription.c_str());

    params.input.data = getIRBlobFromOperand(operation.inputs[0], 0);
    params.input.lifeTime = getV1_3_OperandLifeTime(operation.inputs[0]);

    params.outputState.data = getIRBlobFromOperand(operation.inputs[18], 18);
    params.outputState.lifeTime = getV1_3_OperandLifeTime(operation.inputs[18]);

    params.cellState.data = getIRBlobFromOperand(operation.inputs[19], 19);
    params.cellState.lifeTime = getV1_3_OperandLifeTime(operation.inputs[19]);

    params.input2inputWeights.data     = getIRBlobFromOperand(operation.inputs[1], 1);
    params.input2inputWeights.lifeTime = getV1_3_OperandLifeTime(operation.inputs[1]);

    params.input2ForgetWeights.data     = getIRBlobFromOperand(operation.inputs[2], 2);
    params.input2ForgetWeights.lifeTime    = getV1_3_OperandLifeTime(operation.inputs[2]);

    params.input2CellWeights.data     = getIRBlobFromOperand(operation.inputs[3], 3);
    params.input2CellWeights.lifeTime     = getV1_3_OperandLifeTime(operation.inputs[3]);

    params.input2OutputWeights.data     = getIRBlobFromOperand(operation.inputs[4], 4);
    params.input2OutputWeights.lifeTime     = getV1_3_OperandLifeTime(operation.inputs[4]);

    params.recurrant2inputWeights.data     = getIRBlobFromOperand(operation.inputs[5], 5);
    params.recurrant2inputWeights.lifeTime     = getV1_3_OperandLifeTime(operation.inputs[5]);

    params.recurrant2ForgetWeights.data     = getIRBlobFromOperand(operation.inputs[6], 6);
    params.recurrant2ForgetWeights.lifeTime     = getV1_3_OperandLifeTime(operation.inputs[6]);

    params.recurrant2CellWeights.data     = getIRBlobFromOperand(operation.inputs[7], 7);
    params.recurrant2CellWeights.lifeTime     = getV1_3_OperandLifeTime(operation.inputs[7]);

    params.recurrant2OutputWeights.data     = getIRBlobFromOperand(operation.inputs[8], 8);
    params.recurrant2OutputWeights.lifeTime     = getV1_3_OperandLifeTime(operation.inputs[8]);

    params.cell2InputWeights.data     = getIRBlobFromOperand(operation.inputs[9], 9);
    params.cell2InputWeights.lifeTime     = getV1_3_OperandLifeTime(operation.inputs[9]);

    params.cell2ForgetWeights.data     = getIRBlobFromOperand(operation.inputs[10], 10);
    params.cell2ForgetWeights.lifeTime     = getV1_3_OperandLifeTime(operation.inputs[10]);

    params.cell2OutputWeights.data  = getIRBlobFromOperand(operation.inputs[11], 11);
    params.cell2OutputWeights.lifeTime  = getV1_3_OperandLifeTime(operation.inputs[11]);

    params.inputGateBias.data = getIRBlobFromOperand(operation.inputs[12], 12);
    params.inputGateBias.lifeTime = getV1_3_OperandLifeTime(operation.inputs[12]);

    params.forgetGateBias.data   = getIRBlobFromOperand(operation.inputs[13], 13);
    params.forgetGateBias.lifeTime   = getV1_3_OperandLifeTime(operation.inputs[13]);

    params.cellBias.data = getIRBlobFromOperand(operation.inputs[14], 14);
    params.cellBias.lifeTime = getV1_3_OperandLifeTime(operation.inputs[14]);

    params.outputGateBias.data    = getIRBlobFromOperand(operation.inputs[15], 15);
    params.outputGateBias.lifeTime    = getV1_3_OperandLifeTime(operation.inputs[15]);

    if (lstmDesc.projectionLayerEnabled) {
        params.projectionWeights.data       = getIRBlobFromOperand(operation.inputs[16], 16);
        params.projectionWeights.lifeTime       = getV1_3_OperandLifeTime(operation.inputs[16]);

        params.projectionBias.data    = getIRBlobFromOperand(operation.inputs[17], 17);
        params.projectionBias.lifeTime    = getV1_3_OperandLifeTime(operation.inputs[17]);
    }

    if (params.useLayerNorm) {
        params.inputLayerNormWeights.data      = GetConstOperandAsTensor(operation.inputs[23], 23);
        params.inputLayerNormWeights.lifeTime = getV1_3_OperandLifeTime(operation.inputs[23]);

        params.forgetLayerNormWeights.data     = GetConstOperandAsTensor(operation.inputs[24], 24);
        params.forgetLayerNormWeights.lifeTime = getV1_3_OperandLifeTime(operation.inputs[24]);

        params.cellLayerNormWeights.data       = GetConstOperandAsTensor(operation.inputs[25], 25);
        params.cellLayerNormWeights.lifeTime = getV1_3_OperandLifeTime(operation.inputs[25]);

        params.outputLayerNormWeights.data     = GetConstOperandAsTensor(operation.inputs[26], 26);
        params.outputLayerNormWeights.lifeTime = getV1_3_OperandLifeTime(operation.inputs[26]);
    }

    params.activationFunction = PARAM_I32(20);

    std::vector<std::string> memoryLayers, inLayers;
    auto outputLayerNames = mBuilderModel->createFullLstm(params, lstmDesc, memoryLayers, inLayers);

    auto addIndexToLayerMap = [&](uint32_t index, std::string name, bool memory, bool input) {
        auto operandDetails = mModel.main.operands[index];
        if (input) {
            if (operandDetails.lifetime == V1_3_OperandLifeTime::SUBGRAPH_INPUT) {
                mInputPorts.emplace(std::make_pair(index, LayerInfo(name, memory, DeviceType::GNA)));
            } else if (static_cast<int>(operandDetails.lifetime) == static_cast<int>(OperandLifeTime::TEMPORARY_VARIABLE)) {
                if (mIntermediateLayerMap.find(index) != mIntermediateLayerMap.end()) {
                    auto& halLayer = mIntermediateLayerMap.at(index);
                    halLayer.setInputNode(name, DeviceType::GNA);
                } else {
                    mIntermediateLayerMap.emplace(std::make_pair(index,
                                                            HalLayerInfo(name, DeviceType::GNA,
                                                                         "", DeviceType::None, memory)));
                }
            }
        } else {
            if (operandDetails.lifetime == V1_3_OperandLifeTime::SUBGRAPH_OUTPUT) {
                mOutputToLayerMap.emplace(std::make_pair(index, LayerInfo(name, memory, DeviceType::GNA)));
            } else if (static_cast<int>(operandDetails.lifetime) == static_cast<int>(OperandLifeTime::TEMPORARY_VARIABLE)) {
                if (mIntermediateLayerMap.find(index) != mIntermediateLayerMap.end()) {
                    auto& halLayer = mIntermediateLayerMap.at(index);
                    halLayer.setOutputNode(name, DeviceType::GNA);
                } else {
                    mIntermediateLayerMap.emplace(std::make_pair(index, HalLayerInfo("", DeviceType::None,
                                                                                    name, DeviceType::GNA, memory)));
                }
            }
        }
    };

    addIndexToLayerMap(operation.outputs[0], outputLayerNames[0], false, false);
    addIndexToLayerMap(operation.outputs[1], outputLayerNames[1], false, false);
    addIndexToLayerMap(operation.outputs[2], outputLayerNames[0], false, false);
    addIndexToLayerMap(operation.inputs[18], memoryLayers[0], true, true);
    addIndexToLayerMap(operation.inputs[19], memoryLayers[1], true, true);

    // if (inLayers.size() > 0) {
    //     VLOG(L1, "LSTM layer has input layer!!.. Adding input");
    //     addIndexToLayerMap(operation.inputs[0], inLayers[0], true, true);
    // } else {
    //     VLOG(L1, "LSTM layer has no input layer!!.. Adding input layer to intermediate layer index");
    //     addIndexToLayerMap(operation.inputs[0], "", true, true);
    // }

    // mOutputToLayerMap.emplace(std::make_pair(operation.outputs[0],
    //                                         LayerInfo(outputLayerNames[0], false)));
    // mOutputToLayerMap.emplace(std::make_pair(operation.outputs[1],
    //                                         LayerInfo(outputLayerNames[1], false)));
    // mOutputToLayerMap.emplace(std::make_pair(operation.outputs[2],
    //                                         LayerInfo(outputLayerNames[0], false)));

    // if (memoryLayers.size() > 0) {
    //     mInputPorts.emplace(std::make_pair(operation.inputs[18], LayerInfo(memoryLayers[0], true)));
    //     mInputPorts.emplace(std::make_pair(operation.inputs[19], LayerInfo(memoryLayers[1], true)));

    //     if (inLayers.size() > 0) {
    //         mInputPorts.emplace(std::make_pair(operation.inputs[0], LayerInfo(inLayers[0], false)));
    //     }
    // }

    if (gnaPluginPtr) {
        gnaPluginPtr->setInputIndices({operation.inputs[18], operation.inputs[19], operation.inputs[0]});
        gnaPluginPtr->setOutputIndices({operation.outputs[0], operation.outputs[1], operation.outputs[2]});
    }

    return true;
}

bool GnaPreparedModel::operationQuantizedLSTM(const Operation& operation)
{
    IRBuilder::LstmLayer::QuantLstmParams  params;

    auto getV1_3_OperandLifeTime = [&](uint32_t idx) {
        const auto op = mModel.main.operands[idx];
        return (int)op.lifetime;
    };

    auto getIRBlobFromOperand = [&](uint32_t idx, uint32_t offset) {
        const auto op = mModel.main.operands[idx];

        auto blob = GetConstOperandAsTensor(idx, offset);
        if (op.lifetime == V1_3_OperandLifeTime::SUBGRAPH_INPUT)
        {
            mOpIndex2BlobMap[idx] = blob;
        }

        return blob;
    };

    IRBuilder::LstmLayer::LstmCellDescription lstmDesc;
    lstmDesc.clippingThresholdCellState     = 0;
    lstmDesc.clippingThresholdProjState     = 0;
    lstmDesc.cifgEnabled                    = false;
    lstmDesc.projectionLayerEnabled         = false;
    lstmDesc.peepholeEnabled                = false;

    lstmDesc.clippingThresholdCellState     = PARAM_FP(24); //(mModel, operation, 24);
    lstmDesc.clippingThresholdProjState     = PARAM_FP(25); // mModel, operation, 25);

    std::string lstmDescription;
    if (isOperandDataNull(operation.inputs[1]) ||
        isOperandDataNull(operation.inputs[5]) ||
        isOperandDataNull(operation.inputs[12])) {
        lstmDescription.append("Cifg");
        lstmDesc.cifgEnabled = true;
    } else {
        lstmDescription.append("noCifg");
    }

    if (!isOperandDataNull(operation.inputs[16]))
    {
        lstmDescription.append("Projection");
        lstmDesc.projectionLayerEnabled = true;
    } else {
        lstmDescription.append("noProjection");
    }

    if (!isOperandDataNull(operation.inputs[9]) ||
        !isOperandDataNull(operation.inputs[10]) ||
        !isOperandDataNull(operation.inputs[11]))
    {
        lstmDescription.append("Peephole");
        lstmDesc.peepholeEnabled = true;
    } else {
        lstmDescription.append("noPeephole");
    }

    params.useLayerNorm = true;
    params.useBatchedLayerNorm = true;

    params.input.data = getIRBlobFromOperand(operation.inputs[0], 0);
    params.input.lifeTime = getV1_3_OperandLifeTime(operation.inputs[0]);

    params.outputState.data = getIRBlobFromOperand(operation.inputs[18], 18);
    params.outputState.lifeTime = getV1_3_OperandLifeTime(operation.inputs[18]);

    params.cellState.data = getIRBlobFromOperand(operation.inputs[19], 19);
    params.cellState.lifeTime = getV1_3_OperandLifeTime(operation.inputs[19]);

    if (!lstmDesc.cifgEnabled) {
        params.input2inputWeights.data     = getIRBlobFromOperand(operation.inputs[1], 1);
        params.input2inputWeights.lifeTime = getV1_3_OperandLifeTime(operation.inputs[1]);

        params.recurrant2inputWeights.data     = getIRBlobFromOperand(operation.inputs[5], 5);
        params.recurrant2inputWeights.lifeTime     = getV1_3_OperandLifeTime(operation.inputs[5]);

        params.inputGateBias.data = getIRBlobFromOperand(operation.inputs[12], 12);
        params.inputGateBias.lifeTime = getV1_3_OperandLifeTime(operation.inputs[12]);
    }

    params.input2ForgetWeights.data     = getIRBlobFromOperand(operation.inputs[2], 2);
    params.input2ForgetWeights.lifeTime    = getV1_3_OperandLifeTime(operation.inputs[2]);

    params.input2CellWeights.data     = getIRBlobFromOperand(operation.inputs[3], 3);
    params.input2CellWeights.lifeTime     = getV1_3_OperandLifeTime(operation.inputs[3]);

    params.input2OutputWeights.data     = getIRBlobFromOperand(operation.inputs[4], 4);
    params.input2OutputWeights.lifeTime     = getV1_3_OperandLifeTime(operation.inputs[4]);

    params.recurrant2ForgetWeights.data     = getIRBlobFromOperand(operation.inputs[6], 6);
    params.recurrant2ForgetWeights.lifeTime     = getV1_3_OperandLifeTime(operation.inputs[6]);

    params.recurrant2CellWeights.data     = getIRBlobFromOperand(operation.inputs[7], 7);
    params.recurrant2CellWeights.lifeTime     = getV1_3_OperandLifeTime(operation.inputs[7]);

    params.recurrant2OutputWeights.data     = getIRBlobFromOperand(operation.inputs[8], 8);
    params.recurrant2OutputWeights.lifeTime     = getV1_3_OperandLifeTime(operation.inputs[8]);

    params.cell2InputWeights.data     = getIRBlobFromOperand(operation.inputs[9], 9);
    params.cell2InputWeights.lifeTime     = getV1_3_OperandLifeTime(operation.inputs[9]);

    params.cell2ForgetWeights.data     = getIRBlobFromOperand(operation.inputs[10], 10);
    params.cell2ForgetWeights.lifeTime     = getV1_3_OperandLifeTime(operation.inputs[10]);

    params.cell2OutputWeights.data  = getIRBlobFromOperand(operation.inputs[11], 11);
    params.cell2OutputWeights.lifeTime  = getV1_3_OperandLifeTime(operation.inputs[11]);

    params.forgetGateBias.data   = getIRBlobFromOperand(operation.inputs[13], 13);
    params.forgetGateBias.lifeTime   = getV1_3_OperandLifeTime(operation.inputs[13]);

    params.cellBias.data = getIRBlobFromOperand(operation.inputs[14], 14);
    params.cellBias.lifeTime = getV1_3_OperandLifeTime(operation.inputs[14]);

    params.outputGateBias.data    = getIRBlobFromOperand(operation.inputs[15], 15);
    params.outputGateBias.lifeTime    = getV1_3_OperandLifeTime(operation.inputs[15]);

    if (lstmDesc.projectionLayerEnabled) {
        params.projectionWeights.data       = getIRBlobFromOperand(operation.inputs[16], 16);
        params.projectionWeights.lifeTime       = getV1_3_OperandLifeTime(operation.inputs[16]);

        params.projectionBias.data    = getIRBlobFromOperand(operation.inputs[17], 17);
        params.projectionBias.lifeTime    = getV1_3_OperandLifeTime(operation.inputs[17]);
    }

    if (params.useLayerNorm) {
        if (!lstmDesc.cifgEnabled) {
            params.inputLayerNormWeights.data      = GetConstOperandAsTensor(operation.inputs[20], 20);
            params.inputLayerNormWeights.lifeTime = getV1_3_OperandLifeTime(operation.inputs[20]);

            params.scaleInputGateLayerNorm.data = GetConstOperandAsTensor(operation.inputs[26], 26);
            params.scaleInputGateLayerNorm.lifeTime = getV1_3_OperandLifeTime(operation.inputs[26]);
        }

        params.forgetLayerNormWeights.data     = GetConstOperandAsTensor(operation.inputs[21], 21);
        params.forgetLayerNormWeights.lifeTime = getV1_3_OperandLifeTime(operation.inputs[21]);

        params.cellLayerNormWeights.data       = GetConstOperandAsTensor(operation.inputs[22], 22);
        params.cellLayerNormWeights.lifeTime = getV1_3_OperandLifeTime(operation.inputs[22]);

        params.outputLayerNormWeights.data     = GetConstOperandAsTensor(operation.inputs[23], 23);
        params.outputLayerNormWeights.lifeTime = getV1_3_OperandLifeTime(operation.inputs[23]);

        params.scaleForgetGateLayerNorm.data = GetConstOperandAsTensor(operation.inputs[27], 27);
        params.scaleForgetGateLayerNorm.lifeTime = getV1_3_OperandLifeTime(operation.inputs[27]);

        params.scaleCellGateLayerNorm.data = GetConstOperandAsTensor(operation.inputs[28], 28);
        params.scaleCellGateLayerNorm.lifeTime = getV1_3_OperandLifeTime(operation.inputs[28]);

        params.scaleOutputGateLayerNorm.data = GetConstOperandAsTensor(operation.inputs[29], 29);
        params.scaleOutputGateLayerNorm.lifeTime = getV1_3_OperandLifeTime(operation.inputs[29]);
    }

    params.zeroPointHiddenLayer = PARAM_I32(30);
    params.scalePointHiddenLayer = PARAM_FP(31);

    std::vector<std::string> memoryLayers, inLayers;
    auto outputLayerNames = mBuilderModel->createFullLstm(params, lstmDesc, memoryLayers, inLayers);

    auto addIndexToLayerMap = [&](uint32_t index, std::string name, bool memory, bool input) {
        auto operandDetails = mModel.main.operands[index];
        if (input) {
            if (operandDetails.lifetime == V1_3_OperandLifeTime::SUBGRAPH_INPUT) {
                mInputPorts.emplace(std::make_pair(index, LayerInfo(name, memory, DeviceType::GNA)));
            } else if (static_cast<int>(operandDetails.lifetime) == static_cast<int>(OperandLifeTime::TEMPORARY_VARIABLE)) {
                if (mIntermediateLayerMap.find(index) != mIntermediateLayerMap.end()) {
                    auto& halLayer = mIntermediateLayerMap.at(index);
                    halLayer.setInputNode(name, DeviceType::GNA);
                } else {
                    mIntermediateLayerMap.emplace(std::make_pair(index,
                                                            HalLayerInfo(name, DeviceType::GNA,
                                                                         "", DeviceType::None, memory)));
                }
            }
        } else {
            if (operandDetails.lifetime == V1_3_OperandLifeTime::SUBGRAPH_OUTPUT) {
                mOutputToLayerMap.emplace(std::make_pair(index, LayerInfo(name, memory, DeviceType::GNA)));
            } else if (static_cast<int>(operandDetails.lifetime) == static_cast<int>(OperandLifeTime::TEMPORARY_VARIABLE)) {
                if (mIntermediateLayerMap.find(index) != mIntermediateLayerMap.end()) {
                    auto& halLayer = mIntermediateLayerMap.at(index);
                    halLayer.setOutputNode(name, DeviceType::GNA);
                } else {
                    mIntermediateLayerMap.emplace(std::make_pair(index, HalLayerInfo("", DeviceType::None,
                                                                                     name, DeviceType::GNA, false)));
                }
            }
        }
    };

    addIndexToLayerMap(operation.outputs[0], outputLayerNames[0], false, false);
    addIndexToLayerMap(operation.outputs[1], outputLayerNames[1], false, false);
    addIndexToLayerMap(operation.outputs[2], outputLayerNames[0], false, false);
    addIndexToLayerMap(operation.inputs[18], memoryLayers[0], true, true);
    addIndexToLayerMap(operation.inputs[19], memoryLayers[1], true, true);

    if (inLayers.size() > 0) {
        addIndexToLayerMap(operation.inputs[0], inLayers[0], false, true);
    } else {
        addIndexToLayerMap(operation.inputs[0], "", false, true);
    }

    if (gnaPluginPtr) {
        gnaPluginPtr->setInputIndices({operation.inputs[18], operation.inputs[19], operation.inputs[0]});
        gnaPluginPtr->setOutputIndices({operation.outputs[0], operation.outputs[1], operation.outputs[2]});
    }

    return true;
}

BaseOp* GnaPreparedModel::operationDequantize(const Operation& operation, bool dummyOp) {
    static int count = 0;
    std::string name = "dequantize-cpu-" + std::to_string(count++);
    uint32_t inputIndex = operation.inputs[0], outIndex = operation.outputs[0];
    dummyOpMap[inputIndex] = dummyOp;
    auto operandDetails = mModel.main.operands[operation.inputs[0]];
    if ( (static_cast<int>(operandDetails.lifetime) == static_cast<int>(OperandLifeTime::CONSTANT_COPY)) ||
         (static_cast<int>(operandDetails.lifetime) == static_cast<int>(OperandLifeTime::CONSTANT_REFERENCE))) {
		VLOG(L1, "Dequantize Op on input of time const !!!!! Reevaluate NW");
        nnAssert(false);
    }

    auto OutOperandDetails = mModel.main.operands[operation.outputs[0]];
    if (static_cast<int>(OutOperandDetails.type) == static_cast<int>(OperandType::TENSOR_FLOAT16)) {
        VLOG(L1, "FLOAT16 output for Dequant not tested.. !!!!");
        nnAssert(false);
    }

    BaseOp* dequantOp = nullptr;
    float   scaleFactor = operandDetails.scale;
    int32_t zp = operandDetails.zeroPoint;

    switch(operandDetails.type) {
        case OperandType::TENSOR_QUANT8_ASYMM:
            dequantOp = new DequantizeOp<uint8_t>(name, scaleFactor, zp, dummyOp);
            break;
        case OperandType::TENSOR_QUANT8_ASYMM_SIGNED:
            dequantOp = new DequantizeOp<int8_t>(name, scaleFactor, zp, dummyOp);
            break;
        case OperandType::TENSOR_QUANT8_SYMM:
            dequantOp = new DequantizeOp<int8_t>(name, scaleFactor, zp, dummyOp);
            break;
        case OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
            dequantOp = new DequantizeOp<int8_t>(name, scaleFactor, zp, dummyOp);
            break;
        default:
            VLOG(L1, "Unsupported tensor type TENSOR_QUANT8_SYMM_PER_CHANNEL in dequantize");
            nnAssert(false);
            break;
    }

    dequantOp->setInputIndex(inputIndex, 0);
    RunTimeOperandInfo& outputOp = mOperands[operation.outputs[0]];
    outputOp.outDataType = DataType::FLOAT32;
    if (static_cast<int>(operandDetails.lifetime) == (int)V1_3_OperandLifeTime::SUBGRAPH_INPUT) {
        dequantOp->setSubgraphInput();
        mInputPorts.emplace(std::make_pair(inputIndex, LayerInfo(name, false, DeviceType::CPU)));
    } else if (static_cast<int>(operandDetails.lifetime) == static_cast<int>(OperandLifeTime::TEMPORARY_VARIABLE)) {
        if (mIntermediateLayerMap.find(inputIndex) != mIntermediateLayerMap.end()) {
            auto& halLayer = mIntermediateLayerMap.at(inputIndex);
            halLayer.setInputNode(name, DeviceType::CPU);
        } else {
            mIntermediateLayerMap.emplace(std::make_pair(inputIndex,
                                                        HalLayerInfo(name, DeviceType::CPU, "", DeviceType::None, false)));
        }
    }

    dequantOp->setOutputIndices({outIndex});
    dequantOp->setInputIndices({inputIndex});

    if (static_cast<int>(OutOperandDetails.lifetime) == (int)V1_3_OperandLifeTime::SUBGRAPH_OUTPUT) {
        mOutputToLayerMap.emplace(std::make_pair(outIndex, LayerInfo(name, false, DeviceType::CPU)));
    } else if (static_cast<int>(OutOperandDetails.lifetime) == static_cast<int>(OperandLifeTime::TEMPORARY_VARIABLE)) {
        if (mIntermediateLayerMap.find(inputIndex) != mIntermediateLayerMap.end()) {
            auto& halLayer = mIntermediateLayerMap.at(inputIndex);
            halLayer.setOutputNode(name, DeviceType::CPU);
        } else {
            mIntermediateLayerMap.emplace(std::make_pair(inputIndex,
                                                        HalLayerInfo("", DeviceType::None, name, DeviceType::CPU, false)));
        }
    }

    return dequantOp;
}

BaseOp* GnaPreparedModel::operationQuantize(const Operation& operation, bool dummyOp) {

    static int count = 0;
    std::string name = "quantize-cpu-" + std::to_string(count++);
    uint32_t inputIndex = operation.inputs[0], outIndex = operation.outputs[0];
    RunTimeOperandInfo& inputOp = mOperands[inputIndex];
    dummyOpMap[inputIndex] = dummyOp;
    auto operandDetails = mModel.main.operands[inputIndex];
    bool isFp16 = false;
    if (operandDetails.type == OperandType::FLOAT16 || operandDetails.type == OperandType::TENSOR_FLOAT16){
        VLOG(L1, "is FP16 Quantize\n");
        isFp16 = true;
    }

    auto OutOperandDetails = mModel.main.operands[outIndex];
    BaseOp* quantOp = nullptr;
    float   scaleFactor = OutOperandDetails.scale;
    int32_t zp = OutOperandDetails.zeroPoint;
    RunTimeOperandInfo& outputOp = mOperands[outIndex];

    switch(OutOperandDetails.type) {
        case OperandType::TENSOR_QUANT8_ASYMM:
            if (isFp16) {
                VLOG(L1, "is Fp16 Quantize \n");
                quantOp = new QuantizeOp<_Float16, uint8_t>(name, scaleFactor, zp, dummyOp);
            }
            else {
                VLOG(L1, "is float Quantize \n");
                quantOp = new QuantizeOp<float, uint8_t>(name, scaleFactor, zp, dummyOp);
            }
            break;
        case OperandType::TENSOR_QUANT8_ASYMM_SIGNED:
            if (isFp16) {
                VLOG(L1, "is Fp16 Quantize \n");
                quantOp = new QuantizeOp<_Float16, int8_t>(name, scaleFactor, zp, dummyOp);
            }
            else {
                VLOG(L1, "is float Quantize \n");
                quantOp = new QuantizeOp<float, int8_t>(name, scaleFactor, zp, dummyOp);
            }
            break;
        default:
            nnAssert(false);
            break;
    }

    switch(static_cast<int>(operandDetails.lifetime)) {
        case static_cast<int>(OperandLifeTime::CONSTANT_COPY): {
            auto buf = const_cast<uint8_t*>(&mModel.operandValues[operandDetails.location.offset]);
            auto len_out = sizeOfData(operandDetails.type, operandDetails.dimensions);
            quantOp->setInputData(inputIndex, buf, len_out);
            break;
        }
        case static_cast<int>(OperandLifeTime::CONSTANT_REFERENCE): {
            auto poolIndex = operandDetails.location.poolIndex;
            auto& r = mPoolInfos[poolIndex];
            auto buf = const_cast<uint8_t*>(r.buffer + operandDetails.location.offset);
            auto len_out = sizeOfData(operandDetails.type, operandDetails.dimensions);
            quantOp->setInputData(inputIndex, buf, len_out);
            break;
        }
        case static_cast<int>(V1_3_OperandLifeTime::SUBGRAPH_INPUT): {
            quantOp->setSubgraphInput();
            mInputPorts.emplace(std::make_pair(inputIndex, LayerInfo(name, false, DeviceType::CPU)));
            break;
        }
        case static_cast<int>(OperandLifeTime::TEMPORARY_VARIABLE): {
            if (mIntermediateLayerMap.find(inputIndex) != mIntermediateLayerMap.end()) {
                auto& halLayer = mIntermediateLayerMap.at(inputIndex);
                halLayer.setInputNode(name, DeviceType::CPU);
            } else {
                mIntermediateLayerMap.emplace(std::make_pair(inputIndex,
                                                            HalLayerInfo(name, DeviceType::CPU, "", DeviceType::None, false)));
            }
            break;
        }
        default:
            break;
    }


    quantOp->setInputIndex(inputIndex, 0);
    quantOp->setOutputIndices({outIndex});
    quantOp->setInputIndices({inputIndex});

    if (static_cast<int>(OutOperandDetails.lifetime) == (int)V1_3_OperandLifeTime::SUBGRAPH_OUTPUT) {
        mOutputToLayerMap.emplace(std::make_pair(outIndex, LayerInfo(name, false, DeviceType::CPU)));
    } else if (static_cast<int>(OutOperandDetails.lifetime) == static_cast<int>(OperandLifeTime::TEMPORARY_VARIABLE)) {
        if (mIntermediateLayerMap.find(outIndex) != mIntermediateLayerMap.end()) {
            auto& halLayer = mIntermediateLayerMap.at(outIndex);
            halLayer.setOutputNode(name, DeviceType::CPU);
        } else {
            mIntermediateLayerMap.emplace(std::make_pair(outIndex,
                                                        HalLayerInfo("", DeviceType::None, name, DeviceType::CPU, false)));
        }
    }

    return quantOp;
}

BaseOp* GnaPreparedModel::operationEmbeddingLookup(const Operation& operation) {
    static int count = 0;
    std::string name_values = "embedding-lookup-values" + std::to_string(count++);
    std::string name_lookup = "embedding-lookup" + std::to_string(count++);

    uint32_t lookupIndex = operation.inputs[0];
    uint32_t outIndex = operation.outputs[0];

    BaseOp* embeddinglookupOp = nullptr;

    uint32_t valuesIndex = operation.inputs[1];

    uint32_t outputIndex = operation.outputs[0];
    RunTimeOperandInfo& valuesOp = mOperands[valuesIndex];
    RunTimeOperandInfo& lookupOp = mOperands[lookupIndex];

    auto OutOperandDetails = mModel.main.operands[outputIndex];


    auto ValuesDetails = mModel.main.operands[valuesIndex];
    auto values_dims = ValuesDetails.dimensions;
    auto LookupDetails = mModel.main.operands[lookupIndex];
    auto lookup_dims = LookupDetails.dimensions;
    switch (ValuesDetails.type) {

        case OperandType::FLOAT32:
        case OperandType::TENSOR_FLOAT32:
                embeddinglookupOp = new EmbeddingLookupOp<float>(name_values, values_dims, lookup_dims, ValuesDetails.type);
                break;
            case OperandType::TENSOR_INT32:
            case OperandType::INT32:
                embeddinglookupOp = new EmbeddingLookupOp<int32_t>(name_values, values_dims, lookup_dims, ValuesDetails.type);
                break;
            case OperandType::TENSOR_QUANT8_ASYMM:
            case OperandType::TENSOR_QUANT8_ASYMM_SIGNED:
            case OperandType::TENSOR_QUANT8_SYMM:
                embeddinglookupOp = new EmbeddingLookupOp<int8_t>(name_values, values_dims, lookup_dims, ValuesDetails.type);
                break;
    }
    embeddinglookupOp->setInputIndex(valuesIndex, 1);
    auto currLayer = LayerInfo(name_values, false, DeviceType::CPU);
    if (static_cast<int>(ValuesDetails.lifetime) == static_cast<int>(OperandLifeTime::CONSTANT_COPY)) {
        auto buf = const_cast<uint8_t*>(&mModel.operandValues[ValuesDetails.location.offset]);
        auto len_out = sizeOfData(ValuesDetails.type, ValuesDetails.dimensions);
        embeddinglookupOp->setInputData(valuesIndex, buf, len_out);
    }
    else if (static_cast<int>(ValuesDetails.lifetime) == static_cast<int>(OperandLifeTime::CONSTANT_REFERENCE)) {
            auto poolIndex = ValuesDetails.location.poolIndex;
            auto& r = mPoolInfos[poolIndex];
            auto buf = const_cast<uint8_t*>(r.buffer + ValuesDetails.location.offset);
            auto len_out = sizeOfData(ValuesDetails.type, ValuesDetails.dimensions);
            embeddinglookupOp->setInputData(valuesIndex, buf, len_out);
    }
    else if (static_cast<int>(ValuesDetails.lifetime) == (int)V1_3_OperandLifeTime::SUBGRAPH_INPUT) {
        embeddinglookupOp->setSubgraphInput();
        mInputPorts.emplace(std::make_pair(valuesIndex, currLayer));
    } else if(static_cast<int>(ValuesDetails.lifetime) == static_cast<int>(OperandLifeTime::TEMPORARY_VARIABLE)) {
        if (mIntermediateLayerMap.find(valuesIndex) != mIntermediateLayerMap.end()) {
            auto& halLayer = mIntermediateLayerMap.at(valuesIndex);
            halLayer.setInputNode(name_values, DeviceType::CPU);
        } else {
            mIntermediateLayerMap.emplace(std::make_pair(valuesIndex,
                                                        HalLayerInfo(name_values, DeviceType::CPU, "", DeviceType::None, false)));
        }
    }

    embeddinglookupOp->setInputIndex(lookupIndex, 0);
    currLayer = LayerInfo(name_lookup, false, DeviceType::CPU);

    if (static_cast<int>(LookupDetails.lifetime) == static_cast<int>(OperandLifeTime::CONSTANT_COPY)) {
        auto buf = const_cast<uint8_t*>(&mModel.operandValues[LookupDetails.location.offset]);
        auto len_out = sizeOfData(LookupDetails.type, LookupDetails.dimensions);
        embeddinglookupOp->setInputData(lookupIndex, buf, len_out);
    } else if (static_cast<int>(LookupDetails.lifetime) == static_cast<int>(OperandLifeTime::CONSTANT_REFERENCE)) {
        auto poolIndex = LookupDetails.location.poolIndex;
        auto& r = mPoolInfos[poolIndex];
        auto buf = const_cast<uint8_t*>(r.buffer + LookupDetails.location.offset);
        auto len_out = sizeOfData(LookupDetails.type, LookupDetails.dimensions);
        embeddinglookupOp->setInputData(lookupIndex, buf, len_out);
    } else if (static_cast<int>(LookupDetails.lifetime) == (int)V1_3_OperandLifeTime::SUBGRAPH_INPUT) {
        embeddinglookupOp->setSubgraphInput();
        mInputPorts.emplace(std::make_pair(lookupIndex, currLayer));
    } else if(static_cast<int>(LookupDetails.lifetime) == static_cast<int>(OperandLifeTime::TEMPORARY_VARIABLE)) {
        if (mIntermediateLayerMap.find(lookupIndex) != mIntermediateLayerMap.end()) {
            auto& halLayer = mIntermediateLayerMap.at(lookupIndex);
            halLayer.setInputNode(name_lookup, DeviceType::CPU);
        } else {
            mIntermediateLayerMap.emplace(std::make_pair(lookupIndex,
                                                        HalLayerInfo(name_lookup, DeviceType::CPU, "", DeviceType::None, false)));
        }
    }

    if (static_cast<int>(OutOperandDetails.lifetime) == (int)V1_3_OperandLifeTime::SUBGRAPH_OUTPUT) {
        mOutputToLayerMap.emplace(std::make_pair(outputIndex, currLayer));
    } else if (static_cast<int>(OutOperandDetails.lifetime) == static_cast<int>(OperandLifeTime::TEMPORARY_VARIABLE)) {
        if (mIntermediateLayerMap.find(outputIndex) != mIntermediateLayerMap.end()) {
            auto& halLayer = mIntermediateLayerMap.at(outputIndex);
            halLayer.setOutputNode(name_lookup, DeviceType::CPU);
        } else {
            mIntermediateLayerMap.emplace(std::make_pair(outputIndex,
                                                        HalLayerInfo("", DeviceType::None, name_lookup, DeviceType::CPU, false)));
        }
    }

    embeddinglookupOp->setInputIndices({lookupIndex, valuesIndex});
    embeddinglookupOp->setOutputIndices({outIndex});
    return embeddinglookupOp;
}

IRBlob::Ptr GnaPreparedModel::GetConstWeightsOperandAsTensor(uint32_t index)
{
    dumpOperand(index);
    const auto op = mModel.main.operands[index];
    bool isQuantInput = false;
    if (op.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED ||
            op.type == OperandType::TENSOR_QUANT8_SYMM ||
            op.type == OperandType::TENSOR_QUANT16_SYMM ) {
        isQuantInput = true;
    }
    uint32_t len;
    const uint8_t *buf = GetOperandMemory(mModel, index, len);

    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        if (buf == nullptr)
            VLOG(L1, "TENSOR_FLOAT32 buf is NULL !!!!!!!!!!!!!!!");

        vec<unsigned int> order;
        Layout layout;
        if (op.dimensions.size() == 4) {
            order = {3,0,1,2};  //IHWO -> OIHW for depth conv
            layout = Layout::OIHW; //weights layout
        }
        else if (op.dimensions.size() == 2) {
            order = {0, 1};
            layout = Layout::NC;
        }
        else {
            order = {0};
            layout = Layout::C;
        }

        TensorDims inputDims;
        if (op.dimensions.size() == 3) {
            auto channel_size = op.dimensions[1] * op.dimensions[2];
            uint32_t op_dimensions_size = 2;
            std::vector<uint32_t> op_dimensions = {op.dimensions[0], channel_size};
            inputDims = toDims(op_dimensions);
        }
        else {
            inputDims = toDims(op.dimensions);
        }

        TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(inputDims, order), layout);
        if (inputDims.size() != 4) {
            InferenceEngine::TBlob<float>::Ptr blob = nullptr;
            if (isQuantInput) {
                blob = std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();
#ifdef PERF_COUNTERS
                if (op.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED) {
                    deQuantize((int8_t*)buf, getNumberOfElements(op.dimensions), op.scale, op.zeroPoint, blob->buffer().as<float*>(), runtimeMetrics);
                } else if (op.type == OperandType::TENSOR_QUANT8_SYMM, 8) {
                    deQuantize((int8_t*)buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>(), runtimeMetrics);
                } else if (op.type == OperandType::TENSOR_QUANT16_SYMM) {
                    deQuantize((int16_t*)buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>(), runtimeMetrics);
                }
#else
                if (op.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED) {
                    deQuantize((int8_t*)buf, getNumberOfElements(op.dimensions), op.scale, op.zeroPoint, blob->buffer().as<float*>());
                } else if (op.type == OperandType::TENSOR_QUANT8_SYMM, 8) {
                    deQuantize((int8_t*)buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>());
                } else if (op.type == OperandType::TENSOR_QUANT16_SYMM) {
                    deQuantize((int16_t*)buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>());
                }
#endif
            }
            else {
                InferenceEngine::TBlob<float>::Ptr blob =
                            std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
                blob->allocate();
            }
            return blob;
        } else {
            InferenceEngine::TBlob<float>::Ptr blob =
                                    std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();

            auto dims_ohwi = inputDims;
            size_t out_depth = dims_ohwi[0];
            size_t in_depth = dims_ohwi[3];
            size_t height = dims_ohwi[1];
            size_t width = dims_ohwi[2];
            size_t offset = 0;
            const float* inputFilter = reinterpret_cast<const float *>(buf);

            //convert OHWI -> OIHW
            //for depth conv need reorder as IOHW since for tflite O is always 1 and IE expects reorder to
            //[in_channels, depth_multiplier, filter_height, filter_width]
            for (size_t i = 0; i < in_depth; i++) {
                for (size_t o = 0; o < out_depth; o++) {
                    for (size_t h = 0; h < height; h++) {
                        for (size_t w = 0; w < width; w++) {
                            //similar to NHWC memory layout
                            size_t offset_ohwi = o*height*width*in_depth +
                                                 h*width*in_depth +
                                                 w*in_depth + i;
                            blob->buffer().as<float*>()[offset++] = inputFilter[offset_ohwi];
                        }
                    }
                }
            }
            return blob;
        }
    } else if (op.type == OperandType::TENSOR_INT32) {
        if (buf == nullptr)
            VLOG(L1, "TENSOR_INT32 buf is NULL !!!!!!!!!!!!!!!");

        TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), Layout::ANY);
        return std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
    } else {
        VLOG(L1, "Do not support const tensors of type ", op.type);
        nnAssert(false);
    }
    return nullptr;
}

IRBlob::Ptr GnaPreparedModel::GetConstOperandAsTensor(int operand_index, int operation_idx)
{
    dumpOperand(operand_index);
    const auto op = mModel.main.operands[operand_index];
    bool isQuantInput = false;
    if (op.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED ||
            op.type == OperandType::TENSOR_QUANT8_SYMM ||
            op.type == OperandType::TENSOR_QUANT16_SYMM ||
            op.type == OperandType::TENSOR_INT32) {
        isQuantInput = true;
    }
    uint32_t len;
    const uint8_t *buf = GetOperandMemory(mModel, operand_index, len);

    VLOG(L1, "GnaPreparedModel:: Operand: index: %d, len: %d, buf: %p", operand_index, len, buf);
    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32 ||
        op.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED ||
        op.type == OperandType::TENSOR_QUANT8_SYMM ||
        op.type == OperandType::TENSOR_QUANT16_SYMM ||
        op.type == OperandType::TENSOR_INT32) {
        vec<unsigned int> order;
        Layout layout;
        if (op.dimensions.size() == 4) {
            order = {0,3,1,2};  //nhwc -> nchw
            layout = Layout::OIHW; //weights layout
        } else if (op.dimensions.size() == 2 || op.dimensions.size() == 3) {
            order = {0, 1};
            layout = Layout::NC;
        } else {
            order = {0};
            layout = Layout::C;
        }

        TensorDims inputDims;
        if (op.dimensions.size() == 3) {
            auto channel_size = op.dimensions[1] * op.dimensions[2];
            uint32_t op_dimensions_size = 2;
            std::vector<uint32_t> op_dimensions = {op.dimensions[0], channel_size};
            inputDims = toDims(op_dimensions);
        } else {
            inputDims = toDims(op.dimensions);
        }

        TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(inputDims, order), layout);
        if (buf == nullptr) {
            VLOG(L1, "TENSOR_FLOAT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob =
                                            std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();
            return blob;
        } else {
            if (inputDims.size() != 4) {
                InferenceEngine::TBlob<float>::Ptr blob = nullptr;
                if (isQuantInput) {
                    blob = std::make_shared<InferenceEngine::TBlob<float>>(td);
                    blob->allocate();
#ifdef PERF_COUNTERS
                    if (op.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED) {
                        deQuantize((int8_t*)buf, getNumberOfElements(op.dimensions), op.scale, op.zeroPoint, blob->buffer().as<float*>(), runtimeMetrics);
                    } else if (op.type == OperandType::TENSOR_QUANT8_SYMM) {
                        deQuantize((int8_t*)buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>(), runtimeMetrics);
                    } else if (op.type == OperandType::TENSOR_QUANT16_SYMM) {
                        deQuantize((int16_t*)buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>(), runtimeMetrics);
                    } else if (op.type == OperandType::TENSOR_INT32) {
                        deQuantize((int32_t*)buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>(), runtimeMetrics);
                    }
#else
                    if (op.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED) {
                        deQuantize((int8_t*)buf, getNumberOfElements(op.dimensions), op.scale, op.zeroPoint, blob->buffer().as<float*>());
                    } else if (op.type == OperandType::TENSOR_QUANT8_SYMM) {
                        deQuantize((int8_t*)buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>());
                    } else if (op.type == OperandType::TENSOR_QUANT16_SYMM) {
                        deQuantize((int16_t*)buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>());
                    } else if (op.type == OperandType::TENSOR_INT32) {
                        deQuantize((int32_t*)buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>());
                    }

#endif
                } else {
                    blob = std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
                    blob->allocate();
                }
                if (op.lifetime == V1_3_OperandLifeTime::CONSTANT_COPY || op.lifetime == V1_3_OperandLifeTime::CONSTANT_REFERENCE) {
                    mModelIRBlobs.push_back(blob);
                    buf = nullptr;
                }
                return blob;
            } else {
                InferenceEngine::TBlob<float>::Ptr blob =
                                    std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();

                auto dims_ohwi = inputDims;
                size_t out_depth = dims_ohwi[0];
                size_t in_depth = dims_ohwi[3];
                size_t height = dims_ohwi[1];
                size_t width = dims_ohwi[2];
                size_t offset = 0;
                const float* inputFilter = reinterpret_cast<const float *>(buf); //OHWI memory layout

                for (size_t o = 0; o < out_depth; o++) {
                    for (size_t i = 0; i < in_depth; i++) {
                        for (size_t h = 0; h < height; h++) {
                            for (size_t w = 0; w < width; w++) {
                                //similar to NHWC memory layout
                                size_t offset_ohwi = o*height*width*in_depth +
                                                     h*width*in_depth +
                                                     w*in_depth + i;
                                blob->buffer().as<float*>()[offset++] = inputFilter[offset_ohwi];
                            }
                        }
                    }
                }
                if (op.lifetime == V1_3_OperandLifeTime::CONSTANT_COPY || op.lifetime == V1_3_OperandLifeTime::CONSTANT_REFERENCE) {
                    mModelIRBlobs.push_back(blob);
                    buf = nullptr;
                }
                return blob;
            }
        }
    }
    else {
        VLOG(L1, "Do not support const tensors of type ", op.type);
        nnAssert(false);
    }
    return nullptr;
}

Blob::Ptr GnaPreparedModel::GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len)
{
    bool isQuantInput = false;
    if (op.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED ||
            op.type == OperandType::TENSOR_QUANT8_SYMM ||
            op.type == OperandType::TENSOR_QUANT16_SYMM ) {
        isQuantInput = true;
    }

    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32 || isQuantInput)  {
        if (op.lifetime == V1_3_OperandLifeTime::SUBGRAPH_INPUT) {
            if (buf == nullptr)
                VLOG(L1, "SUBGRAPH_INPUT buf is NULL !!!!!!!!!!!!!!!");

            vec<unsigned int> order;
            Layout layout;
            if (op.dimensions.size() == 4) {
                order = {0,3,1,2};  //nhwc -> nchw
                layout = Layout::NCHW;
            }
            else if (op.dimensions.size() == 2) {
                order = {0, 1};
                layout = Layout::NC;
            }
            else {
                order = {0}; //(op.dimensions.size() < 2)
                layout = Layout::C;
            }

            auto inputDims = toDims(op.dimensions);
            TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(inputDims, order), layout);
            if (inputDims.size() != 4) {
                InferenceEngine::TBlob<float>::Ptr blob = nullptr;

                if (isQuantInput) {
                    blob = std::make_shared<InferenceEngine::TBlob<float>>(td);
                    blob->allocate();
#ifdef PERF_COUNTERS
                    if (op.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED) {
                        deQuantize((int8_t*)buf, getNumberOfElements(op.dimensions), op.scale, op.zeroPoint, blob->buffer().as<float*>(), runtimeMetrics);
                    } else if (op.type == OperandType::TENSOR_QUANT8_SYMM) {
                        deQuantize((int8_t*)buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>(), runtimeMetrics);
                    } else if (op.type == OperandType::TENSOR_QUANT16_SYMM) {
                        deQuantize((int16_t*)buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>(), runtimeMetrics);
                    }
#else
                    if (op.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED) {
                        deQuantize((int8_t*)buf, getNumberOfElements(op.dimensions), op.scale, op.zeroPoint, blob->buffer().as<float*>());
                    } else if (op.type == OperandType::TENSOR_QUANT8_SYMM) {
                        deQuantize((int8_t*)buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>());
                    } else if (op.type == OperandType::TENSOR_QUANT16_SYMM) {
                        deQuantize((int16_t*)buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>());
                    }
#endif
                } else {
                    blob = std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
                    blob->allocate();
                }
                return blob;
            } else {
                InferenceEngine::TBlob<float>::Ptr blob =
                                            std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();

                auto dims_nhwc = inputDims; //toDims(op.dimensions);
                size_t batch = dims_nhwc[0];
                size_t in_depth = dims_nhwc[3]; //channels
                size_t height = dims_nhwc[1];
                size_t width = dims_nhwc[2];
                size_t offset = 0;
                const float* input = reinterpret_cast<const float *>(buf); //OHWI memory layout

                //convert NHWC -> NCHW
                for (size_t b = 0; b < batch; b++) {
                    for (size_t i = 0; i < in_depth; i++) {
                        for (size_t h = 0; h < height; h++) {
                            for (size_t w = 0; w < width; w++) {
                                //similar to NHWC memory layout
                                size_t offset_nhwc = b*height*width*in_depth +
                                                     h*width*in_depth +
                                                     w*in_depth + i;
                                blob->buffer().as<float*>()[offset++] = input[offset_nhwc];
                            }
                        }
                    }
                }
                return blob;
            }
        } else if (op.lifetime == V1_3_OperandLifeTime::SUBGRAPH_OUTPUT) {
            if (buf == nullptr)
                VLOG(L1, "SUBGRAPH_OUTPUT buf is NULL !!!!!!!!!!!!!!!");

            vec<unsigned int> order;
            Layout layout;
            if (op.dimensions.size() == 4) {
                layout = Layout::NHWC;
            } else if (op.dimensions.size() == 2) {
                layout = Layout::NC;
            } else {
                layout = Layout::C;
            }

            TensorDesc td(InferenceEngine::Precision::FP32, toDims(op.dimensions), layout); //nhwc
            InferenceEngine::TBlob<float>::Ptr blob =
                            InferenceEngine::make_shared_blob<float>(td, (float *)buf, len);
            return blob;
        }
    } else if (op.type == OperandType::TENSOR_INT32) {
        TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), Layout::ANY);
        return std::make_shared<InferenceEngine::TBlob<int32_t>>(td, (int32_t *)buf, len);
    } else {
        VLOG(L1, "not supporting const tensors of type ", op.type);
        nnAssert(false);
    }
    return nullptr;
}

}
}
}
}
