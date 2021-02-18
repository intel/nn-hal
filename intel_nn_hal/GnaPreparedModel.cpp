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
    VLOG(L1, "initialize Input");

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
            nnAssert("Cannot set data to non-input layers during infer request");
        }
    }

    for (auto i =0; i < mModel.main.outputIndexes.size(); i++) {
        mModelOutputIndices.emplace_back(mModel.main.outputIndexes[i]);
    }
}

void GnaPreparedModel::initializeInput(std::vector<uint32_t>& indexVec) {
    VLOG(L1, "initialize Input");

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

bool GnaPreparedModel::finalizeOutput(/*RunTimeOperandInfo* output */) {
    VLOG(L1, "finalize Output");
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

bool GnaPreparedModel::initialize(const hidl_vec<hidl_handle>& modelCache, const HidlToken& token) {
    VLOG(L1, "initialize");
    bool success = false;

    // TODO: Remove this hack to identify the nw based on operations
    int lstmCount = 0;
    isDecoderNw = false;
    for (auto op: mModel.main.operations) {
        if (op.type == OperationType::FULLY_CONNECTED) {
            isDecoderNw = true;
            modelNameStr = "Decoder";
        } else if (op.type == OperationType::QUANTIZED_LSTM) {
            lstmCount++;
        }
    }

    if (!isDecoderNw) {
        if (lstmCount > 3) {
            isEnc1Nw = true;
            modelNameStr = "Encoder1";
        } else {
            isEnc0Nw = true;
            modelNameStr = "Encoder0";
        }
    }

    // Check operation supoorted or not, user may not call getOpertionSupported()
    for (const auto& operation : mModel.main.operations) {
        success = isOperationSupported(operation, mModel, mTargetDevice);
        dumpOperationSupport(operation, success);
        if (!success) {
            VLOG(L1, "get unsupported operation in initialize()");
            return false;
        }
    }

    success = setRunTimePoolInfosFromHidlMemories(&mPoolInfos, mModel.pools);
    if (!success) {
        VLOG(L1, "setRunTimePoolInfosFromHidlMemories failed.");
        return false;
    }

    success = initializeRunTimeOperandInfo();
    if (!success) {
        VLOG(L1, "initializeRunTimeOperandInfo failed.");
        return false;
    }

    mBuilderModel = new IRBuilder::ModelBuilder();
    mBuilderModel->initializeBuilder();

    time_point irbuild_start = now();
    for (const auto& operation : mModel.main.operations) {
        VLOG(L1, "get operation %d ready to add", operation.type);
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
            default:
                VLOG(L1, "unsupported operation %d", operation.type);
                return false;
        }
        if (success == false) {
            VLOG(L1, "failed to convert operation %d", operation.type);
            return false;
        }
    }

    initializeInput();

    auto network = mBuilderModel->convertBuilder();
#ifdef PERF_COUNTERS
    time_point irbuild_end = now();
    runtimeMetrics.irBuild_time = (double(millisecondsDuration(irbuild_end, irbuild_start)));
    gnaPluginPtr = new GnaNetwork(network, "GNA");
    InferenceEngine::CNNNetwork passed_network({network});
    gnaPluginPtr->loadNetwork(passed_network, isDecoderNw);
    time_point gnabuild_end = now();
    runtimeMetrics.nw_load_time = (double(millisecondsDuration(gnabuild_end, irbuild_end)));
#else
    gnaPluginPtr = new GnaNetwork(network, "GNA");
    InferenceEngine::CNNNetwork passed_network({network});
    gnaPluginPtr->loadNetwork(passed_network, isDecoderNw);
#endif
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
            writeBits(static_cast<uint32_t>(sizeof(iter.second.size() + 1)), dataCacheFd);
            writeBits(strlen(iter.second.c_str()) + 1, dataCacheFd);
            writeNBytes(iter.second.c_str(), strlen(iter.second.c_str()) + 1 , dataCacheFd);
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

void GnaPreparedModel::asyncExecute(const V1_0_Request& request, MeasureTiming measure, time_point driverStart,
                          const sp<V1_0::IExecutionCallback>& cb) {
    VLOG(L1, "Begin to executeSynchronously");
#ifdef PERF_COUNTERS
    runtimeMetrics.infer_calls++;
#endif
    time_point driverEnd, deviceStart, deviceEnd;
    if (measure == MeasureTiming::YES) deviceStart = now();

    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!setRunTimePoolInfosFromHidlMemories(&requestPoolInfos, request.pools)) {
        cb->notify(V1_0_ErrorStatus::GENERAL_FAILURE);
        return;
    }

    // TODO: We need to change this to add more outputs
    // We are filling outputdata for only 1 output buffer for decoder
    //hidl_vec<OutputShape> outputShapes(request.outputs.size());
    hidl_vec<OutputShape> outputShapes(request.outputs.size());

    auto getBlobFromMemoryPool = [&, this](uint32_t index) {
        RunTimeOperandInfo& operand = mOperands[mModelInputIndices[index]];
        const RequestArgument& arg = request.inputs[index];
        auto poolIndex = arg.location.poolIndex;
        nnAssert(poolIndex < requestPoolInfos.size());
        auto& r = requestPoolInfos[poolIndex];

        if (arg.dimensions.size() > 0) {
                // It's the responsibility of the caller to validate that
                // from.dimensions only modifies the dimensions that were
                // unspecified in the model.main.  That's the case in SampleDriver.cpp
                // with the call to validateRequest().
                operand.dimensions = arg.dimensions;
        }

        operand.buffer = r.buffer + arg.location.offset;
        operand.length = arg.location.length;

        return GetInOutOperandAsBlob(operand,
                                     const_cast<uint8_t*>(r.buffer + arg.location.offset),
                                     operand.length);
    };

    auto copyDataToLayer = [&, this](uint32_t index) {
        auto srcBlob = getBlobFromMemoryPool(index);
        auto iter = mOpIndex2BlobMap.find(mModel.main.inputIndexes[index]);
        if (iter != mOpIndex2BlobMap.end()) {
            auto layerId = mBuilderModel->check4LayerData(iter->second);
            if (layerId != -1)
                mBuilderModel->setLayerData(srcBlob, layerId, iter->second);
        } else {
            ALOGE("Failed to layer for index:", index);
        }
    };

    for (auto index : mlayerInputIndices) {
        auto inputIndex = mModelInputIndices[index];
        auto srcBlob = getBlobFromMemoryPool(index);

        auto iter = mInputPorts.find(inputIndex);
        if (iter != mInputPorts.end()) {
            std::string layerName = iter->second.layerName;

            if (iter->second.memoryLayer) {
                gnaPluginPtr->setMemoryState(layerName, srcBlob);
            } else {
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
                uint8_t* dest = destBlob->buffer().as<uint8_t*>();
                uint8_t* src = srcBlob->buffer().as<uint8_t*>();
                std::memcpy(dest, src, srcBlob->byteSize());
            }
        } else {
            ALOGE("Index:%d not found in input layers map", inputIndex);
        }
    }
    VLOG(L1, "Run");

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

    auto reqOutputs = request.outputs;
    for (auto i =0; i < mModelOutputIndices.size(); i++) {
        auto index = mModelOutputIndices[i];
        RunTimeOperandInfo& operand = mOperands[index];
        const RequestArgument& arg = reqOutputs[i];
        auto poolIndex = arg.location.poolIndex;
        nnAssert(poolIndex < requestPoolInfos.size());
        auto& r = requestPoolInfos[poolIndex];

        void* destPtr = r.buffer + arg.location.offset;

        // Get the name of the layer from which we want to copy the data
        auto elementIdx = mOutputToLayerMap.find(index);
        if (elementIdx != mOutputToLayerMap.end()) {
            auto layerName = elementIdx->second;

            // Check if the layername is present in the output map
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
            }
        } else {
            VLOG(L1, "could not find index:%d in output map", index);
        }
    }

    VLOG(L1, "update shared memories");
    for (auto runtimeInfo : requestPoolInfos) {
        runtimeInfo.update();
    }

    for (auto runtimeInfo : requestPoolInfos) {
        runtimeInfo.unmap_mem();
    }

    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(deviceEnd, deviceStart))};
        VLOG(L1, "Driver::asyncExecute timing = %s", toString(timing).c_str());
        cb->notify(V1_0_ErrorStatus::NONE);
    } else {
        VLOG(L1, "MeasureTiming - No. Returning with no error");
        cb->notify(V1_0_ErrorStatus::NONE);
    }
}

// TODO: call the same asyncExecute function as above
Return<V1_0_ErrorStatus> GnaPreparedModel::executeBase(const V1_0_Request& request, MeasureTiming measure,
                                               const sp<V1_0::IExecutionCallback>& callback) {
    VLOG(L1, "executebase");

    time_point driverStart;
    if (measure == MeasureTiming::YES) driverStart = now();

    if (callback.get() == nullptr) {
        ALOGE("invalid callback passed to execute");
        return V1_0_ErrorStatus::INVALID_ARGUMENT;
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

    return V1_0_ErrorStatus::NONE;
}

bool GnaPreparedModel::operationFullyConnected(const Operation& operation) {
    VLOG(L1, "OperationType::FULLY_CONNECTED");
    /**
     * Denotes a fully (densely) connected layer, which connects all elements
     * in the input tensor with each element in the output tensor.
     *
     * This layer implements the operation:
     *
     *     outputs = activation(inputs * weights’ + bias)
     *
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4.
     *
     * Inputs:
     * * 0: A tensor of at least rank 2, specifying the input. If rank is
     *      greater than 2, then it gets flattened to a 2-D Tensor. The
     *      (flattened) 2-D Tensor is reshaped (if necessary) to
     *      [batch_size, input_size], where "input_size" corresponds to the
     *      number of inputs to the layer, matching the second dimension of
     *      weights, and "batch_size" is calculated by dividing the number of
     *      elements by "input_size".
     * * 1: A 2-D tensor, specifying the weights, of shape
     *      [num_units, input_size], where "num_units" corresponds to the number
     *      of output nodes.
     * * 2: A 1-D tensor, of shape [num_units], specifying the bias. For input
     *      tensor of {@link OperandType::TENSOR_FLOAT32}, the bias should
     *      also be of {@link OperandType::TENSOR_FLOAT32}. For input tensor
     *      of {@link OperandType::TENSOR_QUANT8_ASYMM}, the bias should be
     *      of {@link OperandType::TENSOR_INT32}, with zeroPoint of 0 and
     *      bias_scale == input_scale * filter_scale.
     * * 3: An {@link OperandType::INT32} scalar, and has to be one of the
     *      {@link FusedActivationFunc} values. Specifies the activation to
     *      invoke on the result.
     *
     * Outputs:
     * * 0: The output tensor, of shape [batch_size, num_units]. For output
     *      tensor of {@link OperandType::TENSOR_QUANT8_ASYMM}, the following
     *      condition must be satisfied:
     *      output_scale > input_scale * filter_scale.

    FULLY_CONNECTED = 9,
     */

    auto getV1_0_OperandLifeTime = [&](uint32_t idx) {
        const auto op = mModel.main.operands[idx];
        return (int)op.lifetime;
    };

    auto getIRBlobFromOperand = [&](uint32_t idx, uint32_t offset) {
        const auto op = mModel.main.operands[idx];

        auto blob = GetConstOperandAsTensor(idx, offset);
        if (op.lifetime == V1_0_OperandLifeTime::SUBGRAPH_INPUT)
        {
            mOpIndex2BlobMap[idx] = blob;
            VLOG(L1, "blob idx=%d (model_input) ptr=%p", idx, blob.get());
        }

        return blob;
    };

    // for (auto i=0; i < operation.inputs.size(); i++) {
    //     auto idx = operation.inputs[i];
    //     const auto op = mModel.main.operands[idx];
    //     VLOG(L1, "idx=%d lifetime=%d", idx, op.lifetime);
    // }

    // for (auto i=0; i < operation.outputs.size(); i++) {
    //     VLOG(L1, "output index = %d", operation.outputs[i]);
    // }

    IRBuilder::BuilderFCLayer::FCParams params;
    auto input = getIRBlobFromOperand(operation.inputs[0], 0);

    params.weights.data = getIRBlobFromOperand(operation.inputs[1], 1);
    params.weights.lifeTime = getV1_0_OperandLifeTime(operation.inputs[1]);

    params.bias.data = getIRBlobFromOperand(operation.inputs[2], 2);
    params.bias.lifeTime = getV1_0_OperandLifeTime(operation.inputs[2]);

    auto inputDims = input->getTensorDesc().getDims();
    for (auto i = 0; i < inputDims.size(); i++) VLOG(L1, "input dims[%d] = %d ", i, inputDims[i]);

    auto weightsDims = params.weights.data->getTensorDesc().getDims();
    for (auto i = 0; i < weightsDims.size(); i++)
        VLOG(L1, "weights dims[%d] = %d ", i, weightsDims[i]);

    auto biasDims = params.bias.data->getTensorDesc().getDims();

    // input is [batch_size, input_size], weights is [num_unit, input_size]
    // nnAssert(inputDims[1] == weightsDims[1]);

    nnAssert(inputDims.size() >= 2);
    nnAssert(weightsDims.size() == 2);
    uint32_t numInputElements = sizeOfTensor(inputDims);
    uint32_t num_units = weightsDims[0];
    uint32_t input_size = weightsDims[1];
    uint32_t batch_size = numInputElements / input_size;
    nnAssert(biasDims[0] == num_units);
    nnAssert(input_size * batch_size == numInputElements);

    const auto newInputDims = input->getTensorDesc().getDims();


    if (mBuilderModel == nullptr) {
        VLOG(L1, "mBuilder = nullptr !!!");
        // ASSERT
    }

    std::vector<std::string> inLayers;
    mPorts[operation.outputs[0]] = mBuilderModel->createFC(params, input, inLayers);

    if (inLayers.size() != 0) {
        mInputPorts.emplace(std::make_pair(operation.inputs[0], LayerInfo(inLayers[0], false)));
    }

    /*
        //FIX ME : Work around since input size indims[0] != output nodes (wdims[0])
        auto dims = permuteDims(weights->getTensorDesc().getDims(), {0, 1});
        dims[0] = indims[0];
        weights->getTensorDesc().setDims(dims);
        //WA end
    */

    //builder_FC_create()

    //auto out = weights * input + bias;

    //mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(3));

    VLOG(L1, "----------------------------------------------");
    VLOGDIMS(L1, inputDims, "inputs dims");
    VLOGDIMS(L1, newInputDims, "newInput dims");
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

    auto getV1_0_OperandLifeTime = [&](uint32_t idx) {
        const auto op = mModel.main.operands[idx];
        return (int)op.lifetime;
    };

    auto getIRBlobFromOperand = [&](uint32_t idx, uint32_t offset) {
        const auto op = mModel.main.operands[idx];

        auto blob = GetConstOperandAsTensor(idx, offset);
        if (op.lifetime == V1_0_OperandLifeTime::SUBGRAPH_INPUT)
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
    params.input.lifeTime = getV1_0_OperandLifeTime(operation.inputs[0]);

    params.outputState.data = getIRBlobFromOperand(operation.inputs[18], 18);
    params.outputState.lifeTime = getV1_0_OperandLifeTime(operation.inputs[18]);

    params.cellState.data = getIRBlobFromOperand(operation.inputs[19], 19);
    params.cellState.lifeTime = getV1_0_OperandLifeTime(operation.inputs[19]);

    params.input2inputWeights.data     = getIRBlobFromOperand(operation.inputs[1], 1);
    params.input2inputWeights.lifeTime = getV1_0_OperandLifeTime(operation.inputs[1]);

    params.input2ForgetWeights.data     = getIRBlobFromOperand(operation.inputs[2], 2);
    params.input2ForgetWeights.lifeTime    = getV1_0_OperandLifeTime(operation.inputs[2]);

    params.input2CellWeights.data     = getIRBlobFromOperand(operation.inputs[3], 3);
    params.input2CellWeights.lifeTime     = getV1_0_OperandLifeTime(operation.inputs[3]);

    params.input2OutputWeights.data     = getIRBlobFromOperand(operation.inputs[4], 4);
    params.input2OutputWeights.lifeTime     = getV1_0_OperandLifeTime(operation.inputs[4]);

    params.recurrant2inputWeights.data     = getIRBlobFromOperand(operation.inputs[5], 5);
    params.recurrant2inputWeights.lifeTime     = getV1_0_OperandLifeTime(operation.inputs[5]);

    params.recurrant2ForgetWeights.data     = getIRBlobFromOperand(operation.inputs[6], 6);
    params.recurrant2ForgetWeights.lifeTime     = getV1_0_OperandLifeTime(operation.inputs[6]);

    params.recurrant2CellWeights.data     = getIRBlobFromOperand(operation.inputs[7], 7);
    params.recurrant2CellWeights.lifeTime     = getV1_0_OperandLifeTime(operation.inputs[7]);

    params.recurrant2OutputWeights.data     = getIRBlobFromOperand(operation.inputs[8], 8);
    params.recurrant2OutputWeights.lifeTime     = getV1_0_OperandLifeTime(operation.inputs[8]);

    params.cell2InputWeights.data     = getIRBlobFromOperand(operation.inputs[9], 9);
    params.cell2InputWeights.lifeTime     = getV1_0_OperandLifeTime(operation.inputs[9]);

    params.cell2ForgetWeights.data     = getIRBlobFromOperand(operation.inputs[10], 10);
    params.cell2ForgetWeights.lifeTime     = getV1_0_OperandLifeTime(operation.inputs[10]);

    params.cell2OutputWeights.data  = getIRBlobFromOperand(operation.inputs[11], 11);
    params.cell2OutputWeights.lifeTime  = getV1_0_OperandLifeTime(operation.inputs[11]);

    params.inputGateBias.data = getIRBlobFromOperand(operation.inputs[12], 12);
    params.inputGateBias.lifeTime = getV1_0_OperandLifeTime(operation.inputs[12]);

    params.forgetGateBias.data   = getIRBlobFromOperand(operation.inputs[13], 13);
    params.forgetGateBias.lifeTime   = getV1_0_OperandLifeTime(operation.inputs[13]);

    params.cellBias.data = getIRBlobFromOperand(operation.inputs[14], 14);
    params.cellBias.lifeTime = getV1_0_OperandLifeTime(operation.inputs[14]);

    params.outputGateBias.data    = getIRBlobFromOperand(operation.inputs[15], 15);
    params.outputGateBias.lifeTime    = getV1_0_OperandLifeTime(operation.inputs[15]);

    if (lstmDesc.projectionLayerEnabled) {
        params.projectionWeights.data       = getIRBlobFromOperand(operation.inputs[16], 16);
        params.projectionWeights.lifeTime       = getV1_0_OperandLifeTime(operation.inputs[16]);

        params.projectionBias.data    = getIRBlobFromOperand(operation.inputs[17], 17);
        params.projectionBias.lifeTime    = getV1_0_OperandLifeTime(operation.inputs[17]);
    }

    if (params.useLayerNorm) {
        params.inputLayerNormWeights.data      = GetConstOperandAsTensor(operation.inputs[23], 23);
        params.inputLayerNormWeights.lifeTime = getV1_0_OperandLifeTime(operation.inputs[23]);

        params.forgetLayerNormWeights.data     = GetConstOperandAsTensor(operation.inputs[24], 24);
        params.forgetLayerNormWeights.lifeTime = getV1_0_OperandLifeTime(operation.inputs[24]);

        params.cellLayerNormWeights.data       = GetConstOperandAsTensor(operation.inputs[25], 25);
        params.cellLayerNormWeights.lifeTime = getV1_0_OperandLifeTime(operation.inputs[25]);

        params.outputLayerNormWeights.data     = GetConstOperandAsTensor(operation.inputs[26], 26);
        params.outputLayerNormWeights.lifeTime = getV1_0_OperandLifeTime(operation.inputs[26]);
    }

    params.activationFunction = PARAM_I32(20);

    std::vector<std::string> memoryLayers, inLayers;
    auto outputLayerNames = mBuilderModel->createFullLstm(params, lstmDesc, memoryLayers, inLayers);

    mOutputToLayerMap[operation.outputs[0]] = outputLayerNames[0];
    mOutputToLayerMap[operation.outputs[1]] = outputLayerNames[1];
    mOutputToLayerMap[operation.outputs[2]] = outputLayerNames[0];

    if (memoryLayers.size() > 0) {
        mInputPorts.emplace(std::make_pair(operation.inputs[18], LayerInfo(memoryLayers[0], true)));
        mInputPorts.emplace(std::make_pair(operation.inputs[19], LayerInfo(memoryLayers[1], true)));

        if (inLayers.size() > 0) {
            mInputPorts.emplace(std::make_pair(operation.inputs[0], LayerInfo(inLayers[0], false)));
        }
    }

    return true;
}

bool GnaPreparedModel::operationQuantizedLSTM(const Operation& operation)
{
    IRBuilder::LstmLayer::QuantLstmParams  params;

    auto getV1_0_OperandLifeTime = [&](uint32_t idx) {
        const auto op = mModel.main.operands[idx];
        return (int)op.lifetime;
    };

    auto getIRBlobFromOperand = [&](uint32_t idx, uint32_t offset) {
        const auto op = mModel.main.operands[idx];

        auto blob = GetConstOperandAsTensor(idx, offset);
        if (op.lifetime == V1_0_OperandLifeTime::SUBGRAPH_INPUT)
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


    VLOG(L1, "Lstm cell description %s", lstmDescription.c_str());

    params.input.data = getIRBlobFromOperand(operation.inputs[0], 0);
    params.input.lifeTime = getV1_0_OperandLifeTime(operation.inputs[0]);

    params.outputState.data = getIRBlobFromOperand(operation.inputs[18], 18);
    params.outputState.lifeTime = getV1_0_OperandLifeTime(operation.inputs[18]);

    params.cellState.data = getIRBlobFromOperand(operation.inputs[19], 19);
    params.cellState.lifeTime = getV1_0_OperandLifeTime(operation.inputs[19]);

    params.input2inputWeights.data     = getIRBlobFromOperand(operation.inputs[1], 1);
    params.input2inputWeights.lifeTime = getV1_0_OperandLifeTime(operation.inputs[1]);

    params.input2ForgetWeights.data     = getIRBlobFromOperand(operation.inputs[2], 2);
    params.input2ForgetWeights.lifeTime    = getV1_0_OperandLifeTime(operation.inputs[2]);

    params.input2CellWeights.data     = getIRBlobFromOperand(operation.inputs[3], 3);
    params.input2CellWeights.lifeTime     = getV1_0_OperandLifeTime(operation.inputs[3]);

    params.input2OutputWeights.data     = getIRBlobFromOperand(operation.inputs[4], 4);
    params.input2OutputWeights.lifeTime     = getV1_0_OperandLifeTime(operation.inputs[4]);

    params.recurrant2inputWeights.data     = getIRBlobFromOperand(operation.inputs[5], 5);
    params.recurrant2inputWeights.lifeTime     = getV1_0_OperandLifeTime(operation.inputs[5]);

    params.recurrant2ForgetWeights.data     = getIRBlobFromOperand(operation.inputs[6], 6);
    params.recurrant2ForgetWeights.lifeTime     = getV1_0_OperandLifeTime(operation.inputs[6]);

    params.recurrant2CellWeights.data     = getIRBlobFromOperand(operation.inputs[7], 7);
    params.recurrant2CellWeights.lifeTime     = getV1_0_OperandLifeTime(operation.inputs[7]);

    params.recurrant2OutputWeights.data     = getIRBlobFromOperand(operation.inputs[8], 8);
    params.recurrant2OutputWeights.lifeTime     = getV1_0_OperandLifeTime(operation.inputs[8]);

    params.cell2InputWeights.data     = getIRBlobFromOperand(operation.inputs[9], 9);
    params.cell2InputWeights.lifeTime     = getV1_0_OperandLifeTime(operation.inputs[9]);

    params.cell2ForgetWeights.data     = getIRBlobFromOperand(operation.inputs[10], 10);
    params.cell2ForgetWeights.lifeTime     = getV1_0_OperandLifeTime(operation.inputs[10]);

    params.cell2OutputWeights.data  = getIRBlobFromOperand(operation.inputs[11], 11);
    params.cell2OutputWeights.lifeTime  = getV1_0_OperandLifeTime(operation.inputs[11]);

    params.inputGateBias.data = getIRBlobFromOperand(operation.inputs[12], 12);
    params.inputGateBias.lifeTime = getV1_0_OperandLifeTime(operation.inputs[12]);

    params.forgetGateBias.data   = getIRBlobFromOperand(operation.inputs[13], 13);
    params.forgetGateBias.lifeTime   = getV1_0_OperandLifeTime(operation.inputs[13]);

    params.cellBias.data = getIRBlobFromOperand(operation.inputs[14], 14);
    params.cellBias.lifeTime = getV1_0_OperandLifeTime(operation.inputs[14]);

    params.outputGateBias.data    = getIRBlobFromOperand(operation.inputs[15], 15);
    params.outputGateBias.lifeTime    = getV1_0_OperandLifeTime(operation.inputs[15]);

    if (lstmDesc.projectionLayerEnabled) {
        params.projectionWeights.data       = getIRBlobFromOperand(operation.inputs[16], 16);
        params.projectionWeights.lifeTime       = getV1_0_OperandLifeTime(operation.inputs[16]);

        params.projectionBias.data    = getIRBlobFromOperand(operation.inputs[17], 17);
        params.projectionBias.lifeTime    = getV1_0_OperandLifeTime(operation.inputs[17]);
    }

    if (params.useLayerNorm) {
        params.inputLayerNormWeights.data      = GetConstOperandAsTensor(operation.inputs[20], 20);
        params.inputLayerNormWeights.lifeTime = getV1_0_OperandLifeTime(operation.inputs[20]);

        params.forgetLayerNormWeights.data     = GetConstOperandAsTensor(operation.inputs[21], 21);
        params.forgetLayerNormWeights.lifeTime = getV1_0_OperandLifeTime(operation.inputs[21]);

        params.cellLayerNormWeights.data       = GetConstOperandAsTensor(operation.inputs[22], 22);
        params.cellLayerNormWeights.lifeTime = getV1_0_OperandLifeTime(operation.inputs[22]);

        params.outputLayerNormWeights.data     = GetConstOperandAsTensor(operation.inputs[23], 23);
        params.outputLayerNormWeights.lifeTime = getV1_0_OperandLifeTime(operation.inputs[23]);

        params.scaleInputGateLayerNorm.data = GetConstOperandAsTensor(operation.inputs[26], 26);
        params.scaleInputGateLayerNorm.lifeTime = getV1_0_OperandLifeTime(operation.inputs[26]);

        params.scaleForgetGateLayerNorm.data = GetConstOperandAsTensor(operation.inputs[27], 27);
        params.scaleForgetGateLayerNorm.lifeTime = getV1_0_OperandLifeTime(operation.inputs[27]);

        params.scaleCellGateLayerNorm.data = GetConstOperandAsTensor(operation.inputs[28], 28);
        params.scaleCellGateLayerNorm.lifeTime = getV1_0_OperandLifeTime(operation.inputs[28]);

        params.scaleOutputGateLayerNorm.data = GetConstOperandAsTensor(operation.inputs[29], 29);
        params.scaleOutputGateLayerNorm.lifeTime = getV1_0_OperandLifeTime(operation.inputs[29]);


    }

    params.zeroPointHiddenLayer = PARAM_I32(30);
    params.scalePointHiddenLayer = PARAM_FP(31);

    std::vector<std::string> memoryLayers, inLayers;
    auto outputLayerNames = mBuilderModel->createFullLstm(params, lstmDesc, memoryLayers, inLayers);

    mOutputToLayerMap[operation.outputs[0]] = outputLayerNames[0];
    mOutputToLayerMap[operation.outputs[1]] = outputLayerNames[1];
    mOutputToLayerMap[operation.outputs[2]] = outputLayerNames[0];

    if (memoryLayers.size() > 0) {
        mInputPorts.emplace(std::make_pair(operation.inputs[18], LayerInfo(memoryLayers[0], true)));
        mInputPorts.emplace(std::make_pair(operation.inputs[19], LayerInfo(memoryLayers[1], true)));

        if (inLayers.size() > 0) {
            mInputPorts.emplace(std::make_pair(operation.inputs[0], LayerInfo(inLayers[0], false)));
        }
    }

    return true;
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
        VLOG(L1, "__func__ Quant input");
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
        if (op.lifetime == V1_0_OperandLifeTime::SUBGRAPH_INPUT) {
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
        } else if (op.lifetime == V1_0_OperandLifeTime::SUBGRAPH_OUTPUT) {
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
