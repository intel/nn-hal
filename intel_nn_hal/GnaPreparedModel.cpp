#define LOG_TAG "GnaPreparedModel"

#include "GnaPreparedModel.h"
#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>
#include <fstream>
#include <thread>
#include "ValidateHal.h"
#include "Utils.h"

//#include "ExecutionBurstServer.h"
//#include "OperationsUtils.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

using namespace android::nn;
static const Timing kNoTiming = {.timeOnDevice = UINT64_MAX, .timeInDriver = UINT64_MAX};

template <typename T>
T getOperandConstVal(const Model& model, const Operand& operand) {
    const T* data = reinterpret_cast<const T*>(&model.operandValues[operand.location.offset]);
    return data[0];
}

// TODO: Code duplication
static Return<void> notify(const sp<V1_0::IExecutionCallback>& callback, const ErrorStatus& status,
                           const hidl_vec<OutputShape>&, Timing) {
    return callback->notify(status);
}

static Return<void> notify(const sp<V1_2::IExecutionCallback>& callback, const ErrorStatus& status,
                           const hidl_vec<OutputShape>& outputShapes, Timing timing) {
    return callback->notify_1_2(status, outputShapes, timing);
}

void GnaPreparedModel::initializeInput() {
    VLOG(L1, "initialize Input");

#if 0
    for (auto i : mModel.inputIndexes) {
        int dims_size = mOperands[i].dimensions.size();

        VLOGDIMS(L1, mOperands[i].dimensions, "current operand inpu dims:");

        //auto inputDims = mPorts[i]->getTensorDesc().getDims();

        /* uint32_t nelem = getNumberOfElements(mOperands[i].dimensions);
        auto inputElem = sizeOfTensor(inputDims);
        if (nelem != inputElem) {
            VLOG(L1, "set operand input dims to real input dims\n");
            for (auto j = 0; j < inputDims.size(); j++)
                mOperands[i].dimensions[j] = static_cast<uint32_t>(inputDims[j]);
            mOperands[i].length = sizeOfData(mOperands[i].type, mOperands[i].dimensions);
        } */
    }
#endif
}


bool GnaPreparedModel::finalizeOutput(/*RunTimeOperandInfo* output */) {
    VLOG(L1, "finalize Output");
    for (auto i : mModel.outputIndexes) {
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

bool GnaPreparedModel::initialize() {
    VLOG(L1, "initialize");
    bool success = false;

    // Check operation supoorted or not, user may not call getOpertionSupported()
    for (const auto& operation : mModel.operations) {
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

    for (const auto& operation : mModel.operations) {
        VLOG(L1, "get operation %d ready to add", operation.type);
        dumpOperation(operation);
        switch (operation.type) {

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

    mBuilderModel->addOutputLayer();
    initializeInput();
    finalizeOutput();


    // initialize IE operation input/output ports
    //    convertModel(mNet);

    // debug graph
   /*  mNet.buildNetwork();
    std::fstream dot;
    std::string graphfile("/data/local/graphfile");
    dot.open("/data/local/graph.dot", std::ios::out);
    mNet.save(graphfile);
    mNet.crateDotFile(dot);
    dot.close(); */

    return true;
}

// TODO: Call parent class deinitialize from here
void GnaPreparedModel::deinitialize() {
    VLOG(L1, "deinitialize");
    delete enginePtr;
    enginePtr = nullptr;

    delete mBuilderModel;
    mBuilderModel = nullptr;

    delete gnaPluginPtr;
    gnaPluginPtr = nullptr;

    for (const auto& operand : mOperands) {
        /*        for (const auto& buf : operand.buffer) {
                    VLOG(L1, "free buffer %p of operand %p", buf, &operand);
                    if (buf != nullptr)
                    delete buf;
                }*/
        // VLOG(L1, "free buffer %p of operand %p", operand.buffer, &operand);
        // if (operand.buffer)
        //    delete operand.buffer;
    }
    VLOG(L1, "free engine");
}

void GnaPreparedModel::asyncExecute_lstm(const Request& request, MeasureTiming measure, time_point driverStart,
                          const sp<V1_2::IExecutionCallback>& callback) {
    VLOG(L1, "Begin asyncExecute_lstm");

    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!setRunTimePoolInfosFromHidlMemories(&requestPoolInfos, request.pools)) {
        notify(callback, ErrorStatus::GENERAL_FAILURE, {}, kNoTiming);
        return;
    }

    time_point driverEnd, deviceStart, deviceEnd;
    if (measure == MeasureTiming::YES) deviceStart = now();

        auto inOutData = [this, &requestPoolInfos](const std::vector<uint32_t>& indexes,
                                               const hidl_vec<RequestArgument>& arguments,
                                               bool inputFromRequest, Blob::Ptr& getBlob,
                                               std::vector<OutputPort> mPorts) {
        // do memcpy for input data
        for (size_t i = 0; i < indexes.size(); i++) {

            RunTimeOperandInfo& operand = mOperands[indexes[i]];
            const RequestArgument& arg = arguments[i];
            auto poolIndex = arg.location.poolIndex;
            nnAssert(poolIndex < requestPoolInfos.size());
            auto& r = requestPoolInfos[poolIndex];
            if (arg.dimensions.size() > 0) {
                // It's the responsibility of the caller to validate that
                // from.dimensions only modifies the dimensions that were
                // unspecified in the model.  That's the case in SampleDriver.cpp
                // with the call to validateRequest().
                operand.dimensions = arg.dimensions;
            }
            operand.buffer = r.buffer + arg.location.offset;
            operand.length = arg.location.length;

            if (inputFromRequest) {
                // model/request oputput pointer pass to inference engine input
                // memcpy(operand.buffer, r.buffer + arg.location.offset, operand.length)
                if (i == 0) {
                    // VLOG(L1, "setBlob for mPorts[%d]\n", indexes[i]);
                    getBlob  = GetInOutOperandAsBlob(
                                        operand, const_cast<uint8_t*>(r.buffer + arg.location.offset),
                                        operand.length);  // if not doing memcpy

                } else {
                    if (gnaPluginPtr == nullptr) {
                        auto inputBlob = GetInOutOperandAsBlob(
                                            operand,
                                            const_cast<uint8_t*>(r.buffer + arg.location.offset),
                                            operand.length);  // if not doing memcpy

                        for (auto iter : mOpIndex2BlobMap) {
                            if (iter.first == indexes[i]) {
                                std::cout << "found index id=" << indexes[i]
                                            << "to be waiting for memcpy ptr=" << iter.second.get();
                                int layer_id = mBuilderModel->check4LayerData(iter.second);
                                if( layer_id != -1) {
                                    //VLOG(L1, "also found memcpy blob for id=%d", layer_id);
                                    std::cout << "calling setLayerData" << std::endl;
                                    mBuilderModel->setLayerData(inputBlob, layer_id, iter.second);
                                }
                                else
                                    VLOG(L1, "failed to found memcpy blob for id=%d", layer_id);

                            }
                        }
                    }
                }
            } else {
                if(i == 4) {
                    getBlob = GetInOutOperandAsBlob(operand, 
                                                    const_cast<uint8_t*>(r.buffer + arg.location.offset),
                                                    operand.length);  // if not doing memcpy
                }
            }
        }
    };
    VLOG(L1, "pass request inputs/outputs buffer to network/model respectively");


    Blob::Ptr getInputBlob, getOutputBlob;
    inOutData(mModel.inputIndexes, request.inputs, true, getInputBlob, mPorts);
    inOutData(mModel.outputIndexes, request.outputs, false, getOutputBlob, mPorts);

    for (int i = 0; i < getInputBlob->byteSize()/4 ; i++) {
        float *src = getInputBlob->buffer().as<float*>();
        if ( *(src + i) == 1.0) {
            // VLOG(L1, "inBlob elements %d = %f", i, *(src +i));
            std::cout << "inBlob elements " << i << " = " <<  *(src +i) << std::endl;
        }
    }

    if (gnaPluginPtr == nullptr) {
        auto network = mBuilderModel->convertBuilder();
        //VLOG(L1, "initialize ExecuteNetwork for device %s",
                // InferenceEngine::TargetDeviceInfo::name(mTargetDevice)); // TODO: Fix this line
        gnaPluginPtr = new GnaNetwork(network, "GNA");
        //gnaPluginPtr->prepareInput();
        InferenceEngine::CNNNetwork passed_network({network});

        //CNNNetwork network1((ICNNNetwork*)&(*network));
        gnaPluginPtr->loadNetwork(passed_network);
    }

    std::cout << "setting input blobs" << std::endl;
    auto inputInfoItem = gnaPluginPtr->inputInfo.begin()->second->getInputData();
    //InferenceEngine::InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    //std::string input_name = network.getInputsInfo().begin()->first;
    std::vector<Blob::Ptr> ptrInputBlobs;
    for (auto& input : gnaPluginPtr->inputInfo) {
        //VLOG(L1,"%p", (void*)enginePtr->inferRequest.GetBlob(input.first));
        ptrInputBlobs.push_back(gnaPluginPtr->getInferRequest().GetBlob(input.first));
    }
    float* source = getInputBlob->buffer().as<float*>();
    float* dest = ptrInputBlobs[0]->buffer().as<float*>();

    //std::cout << "In gate blob address is = " << mBuilderModel->mInGateBlob << "size = " << mBuilderModel->mInGateBlob->size() << "\n";

    for (int i = 0; i < getInputBlob->byteSize()/4; i++) {
       *(dest + i) = *(source + i);
    }
    VLOG(L1,"input_0 = %s\n", inputInfoItem->getName().c_str());
    //enginePtr->setBlob(inputInfoItem->getName(), getInputBlob);

    auto outputBlob = gnaPluginPtr->getInferRequest().GetBlob(gnaPluginPtr->outputInfo.begin()->first);
    //enginePtr->setBlob(outputInfoItem->getName(), getOutputBlob);
    VLOG(L1, "Run");

    // auto output = execute.Infer(input).wait();
    gnaPluginPtr->Infer();
    source = outputBlob->buffer().as<float*>();
    dest = getOutputBlob->buffer().as<float*>();

    for (int i = 0; i < outputBlob->byteSize()/4; i++) {
       *(dest + i) = *(source + i);
        VLOG(L1, "output = %f addr = %p\n ",*(dest + i), outputBlob.get());
    }

/*
    std::cout << "After loadNetwork" << std::endl;
    // TODO: Change the logic here
	auto inputInfoItem = gnaPluginPtr->inputInfo.begin()->second->getInputData();
    gnaPluginPtr->setBlob(inputInfoItem->getName(), getInputBlob);

    std::cout << "setblob for output" << std::endl;
    auto outputInfoItem = gnaPluginPtr->outputInfo.begin()->second;
    gnaPluginPtr->setBlob(outputInfoItem->getName(), getOutputBlob);

    std::cout << "infer()" << std::endl;
    gnaPluginPtr->Infer();

    if (measure == MeasureTiming::YES) deviceEnd = now();
*/

    VLOG(L1, "update shared memories");
    for (auto runtimeInfo : requestPoolInfos) {
        runtimeInfo.update();
    }

    Return<void> returned = callback->notify(ErrorStatus::NONE);
    if (!returned.isOk()) {
        ALOGE("hidl callback failed to return properly: %s", returned.description().c_str());
    }

    hidl_vec<OutputShape> outputShapes; // TODO: ??
    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(driverEnd, driverStart))};
        ALOGE("Driver::asyncExecute_lstm timing = %s", toString(timing).c_str());
        returned = notify(callback, ErrorStatus::NONE, outputShapes, timing);
    } else {
        returned = notify(callback, ErrorStatus::NONE, outputShapes, kNoTiming);
    }
    if (!returned.isOk()) {
        ALOGE("hidl callback failed to return properly: %s", returned.description().c_str());
    }
}

// TODO: call the same asyncExecute function as above
Return<ErrorStatus> GnaPreparedModel::executeBase(const Request& request, MeasureTiming measure,
                                               const sp<V1_0::IExecutionCallback>& callback) {
    return PreparedModel::executeBase(request, measure, callback);
}

Return<ErrorStatus> GnaPreparedModel::executeBase_1_2(const Request& request, MeasureTiming measure,
                                                   const sp<V1_2::IExecutionCallback>& callback) {
    VLOG(L1, "executebase");

    time_point driverStart;
    if (measure == MeasureTiming::YES) driverStart = now();

    if (callback.get() == nullptr) {
        ALOGE("invalid callback passed to execute");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!validateRequest(request, mModel)) {
        notify(callback, ErrorStatus::INVALID_ARGUMENT, {}, kNoTiming);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // This thread is intentionally detached because the driver service
    // is expected to live forever.
    std::thread([this, request, measure, driverStart, callback] {
        asyncExecute_lstm(request, measure, driverStart, callback);
    }).detach();

    return ErrorStatus::NONE;
}

Return<void> GnaPreparedModel::executeSynchronously(const Request& request, MeasureTiming measure,
                                                 executeSynchronously_cb cb) {
    VLOG(L1, "Begin to executeSynchronously");

    time_point driverEnd, deviceStart, deviceEnd;
    if (measure == MeasureTiming::YES) deviceStart = now();

    if (!validateRequest(request, mModel)) {
        cb(ErrorStatus::INVALID_ARGUMENT, {}, kNoTiming);
        return Void();
    }

    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!setRunTimePoolInfosFromHidlMemories(&requestPoolInfos, request.pools)) {
        cb(ErrorStatus::GENERAL_FAILURE, {}, kNoTiming);
        return Void();
    }

    // TODO: We need to change this to add more outputs
    // We are filling outputdata for only 1 output buffer for decoder
    //hidl_vec<OutputShape> outputShapes(request.outputs.size());
    hidl_vec<OutputShape> outputShapes(request.outputs.size());

    auto inOutData = [this, &requestPoolInfos](const std::vector<uint32_t>& indexes,
                                            const hidl_vec<RequestArgument>& arguments,
                                            bool inputFromRequest, Blob::Ptr& getBlob,
                                            std::vector<OutputPort> mPorts) {
        // do memcpy for input data
        for (size_t i = 0; i < indexes.size(); i++) {
            RunTimeOperandInfo& operand = mOperands[indexes[i]];
            const RequestArgument& arg = arguments[i];
            auto poolIndex = arg.location.poolIndex;
            nnAssert(poolIndex < requestPoolInfos.size());
            auto& r = requestPoolInfos[poolIndex];
            if (arg.dimensions.size() > 0) {
                // It's the responsibility of the caller to validate that
                // from.dimensions only modifies the dimensions that were
                // unspecified in the model.  That's the case in SampleDriver.cpp
                // with the call to validateRequest().
                operand.dimensions = arg.dimensions;
            }
            operand.buffer = r.buffer + arg.location.offset;
            operand.length = arg.location.length;

            if (inputFromRequest) {
                // model/request oputput pointer pass to inference engine input
                // memcpy(operand.buffer, r.buffer + arg.location.offset, operand.length)
                if (i == 0) {
                    // VLOG(L1, "setBlob for mPorts[%d]\n", indexes[i]);
                    getBlob  = GetInOutOperandAsBlob(
                                        operand, const_cast<uint8_t*>(r.buffer + arg.location.offset),
                                        operand.length);  // if not doing memcpy

                } else {
                    if (gnaPluginPtr == nullptr) {
                        auto inputBlob = GetInOutOperandAsBlob(
                                            operand,
                                            const_cast<uint8_t*>(r.buffer + arg.location.offset),
                                            operand.length);  // if not doing memcpy

                        for (auto iter : mOpIndex2BlobMap) {
                            if (iter.first == indexes[i]) {
                                // VLOG(L1, "found index id=%d to be waiting for memcpy ptr=%p", indexes[i], iter.second.get());
                                std::cout << "found index id=" << indexes[i]
                                            << "to be waiting for memcpy ptr=" << iter.second.get();
                                int layer_id = mBuilderModel->check4LayerData(iter.second);
                                if( layer_id != -1) {
                                    //VLOG(L1, "also found memcpy blob for id=%d", layer_id);
                                    std::cout << "calling setLayerData" << std::endl;
                                    mBuilderModel->setLayerData(inputBlob, layer_id, iter.second);
                                }
                                else
                                    VLOG(L1, "failed to found memcpy blob for id=%d", layer_id);

                            }
                        }
                    }
                }
            } else {
                if(indexes.size() < 6 && i == 4) {
                    getBlob = GetInOutOperandAsBlob(operand, const_cast<uint8_t*>(r.buffer + arg.location.offset),operand.length);  // if not doing memcpy
                }
                else if(indexes.size() > 6 && i == 12) {
                    getBlob = GetInOutOperandAsBlob(operand, const_cast<uint8_t*>(r.buffer + arg.location.offset),operand.length);
                }
            }
        }
    };

    Blob::Ptr getInputBlob, getOutputBlob;
    inOutData(mModel.inputIndexes, request.inputs, true, getInputBlob, mPorts);
    inOutData(mModel.outputIndexes, request.outputs, false, getOutputBlob, mPorts);

    // for (int i = 0; i < getInputBlob->byteSize()/4 ; i++) {
    //     float *src = getInputBlob->buffer().as<float*>();
    //     if ( *(src + i) == 1.0) {
    //         // VLOG(L1, "inBlob elements %d = %f", i, *(src +i));
    //         std::cout << "inBlob elements " << i << " = " <<  *(src +i) << std::endl;
    //     }
    // }

    if (gnaPluginPtr == nullptr) {
        auto network = mBuilderModel->convertBuilder();
        gnaPluginPtr = new GnaNetwork(network, "GNA");
        InferenceEngine::CNNNetwork passed_network({network});
        gnaPluginPtr->loadNetwork(passed_network);
    }

    //auto inputInfoItem = gnaPluginPtr->inputInfo.begin()->second->getInputData();
    //InferenceEngine::InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    //std::string input_name = network.getInputsInfo().begin()->first;
    std::vector<Blob::Ptr> ptrInputBlobs;

    for (auto& input : gnaPluginPtr->inputInfo) {
        //VLOG(L1,"%p", (void*)enginePtr->inferRequest.GetBlob(input.first));
        ptrInputBlobs.push_back(gnaPluginPtr->getInferRequest().GetBlob(input.first));
    }

    // TODO: Error check
    float* source = getInputBlob->buffer().as<float*>();
    float* dest = ptrInputBlobs[0]->buffer().as<float*>();

    //std::cout << "In gate blob address is = " << mBuilderModel->mInGateBlob << "size = " << mBuilderModel->mInGateBlob->size() << "\n";

    for (int i = 0; i < getInputBlob->byteSize()/4; i++) {
       *(dest + i) = *(source + i);
    }
    //VLOG(L1,"input_0 = %s\n", inputInfoItem->getName().c_str());
    //enginePtr->setBlob(inputInfoItem->getName(), getInputBlob);

    auto outputBlob = gnaPluginPtr->getInferRequest().GetBlob(gnaPluginPtr->outputInfo.begin()->first);
    auto outputDimensions = gnaPluginPtr->outputInfo.begin()->second->getDims();

    // Need to fill the outputshapes here
    for (auto i=0; i < 4; i++) {
        outputShapes[i].dimensions = {};
        outputShapes[i].isSufficient = true;
    }

    // TODO: Change this code once we get more than one output from model
    hidl_vec<uint32_t> dimensions;
    for (auto i =0; i < outputDimensions.size(); i++) {
        dimensions[i] = outputDimensions[i]; 
    }
    outputShapes[4].dimensions = dimensions;
    outputShapes[4].isSufficient = true;

    //enginePtr->setBlob(outputInfoItem->getName(), getOutputBlob);
    VLOG(L1, "Run");

    // auto output = execute.Infer(input).wait();
    gnaPluginPtr->Infer();
    source = outputBlob->buffer().as<float*>();
    dest = getOutputBlob->buffer().as<float*>();

    for (int i = 0; i < outputBlob->byteSize()/4; i++) {
       *(dest + i) = *(source + i);
        //VLOG(L1, "output = %f addr = %p\n ",*(dest + i), outputBlob.get());
    }

/*
    std::cout << "After loadNetwork" << std::endl;
    // TODO: Change the logic here
	auto inputInfoItem = gnaPluginPtr->inputInfo.begin()->second->getInputData();
    gnaPluginPtr->setBlob(inputInfoItem->getName(), getInputBlob);

    std::cout << "setblob for output" << std::endl;
    auto outputInfoItem = gnaPluginPtr->outputInfo.begin()->second;
    gnaPluginPtr->setBlob(outputInfoItem->getName(), getOutputBlob);

    std::cout << "infer()" << std::endl;
    gnaPluginPtr->Infer();

    if (measure == MeasureTiming::YES) deviceEnd = now();
*/

    VLOG(L1, "update shared memories");
    for (auto runtimeInfo : requestPoolInfos) {
        runtimeInfo.update();
    }

    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(deviceEnd, deviceStart))};
        ALOGE("Driver::asyncExecute_lstm timing = %s", toString(timing).c_str());
        cb(ErrorStatus::NONE, outputShapes, timing);
    } else {
        VLOG(L1, "MeasureTiming - No. Returning with no error");
        cb(ErrorStatus::NONE, outputShapes, kNoTiming);
    }

    return Void();
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

    auto getOperandLifeTime = [&](uint32_t idx) {
        const auto op = mModel.operands[idx];
        return (int)op.lifetime;
    };

    auto getIRBlobFromOperand = [&](uint32_t idx, uint32_t offset) {
        const auto op = mModel.operands[idx];

        auto blob = GetConstOperandAsTensor(idx, offset);
        if (op.lifetime == OperandLifeTime::MODEL_INPUT)
        {
            mOpIndex2BlobMap[idx] = blob;
            VLOG(L1, "blob idx=%d (model_input) ptr=%p", idx, blob.get());
        }

        return blob;
    };

    // for (auto i=0; i < operation.inputs.size(); i++) {
    //     auto idx = operation.inputs[i];
    //     const auto op = mModel.operands[idx];
    //     VLOG(L1, "idx=%d lifetime=%d", idx, op.lifetime);
    // }

    // for (auto i=0; i < operation.outputs.size(); i++) {
    //     VLOG(L1, "output index = %d", operation.outputs[i]);
    // }

    IRBuilder::BuilderFCLayer::FCParams params;
    auto input = getIRBlobFromOperand(operation.inputs[0], 0);

    params.weights.data = getIRBlobFromOperand(operation.inputs[1], 1);
    params.weights.lifeTime = getOperandLifeTime(operation.inputs[1]);

    params.bias.data = getIRBlobFromOperand(operation.inputs[2], 2);
    params.bias.lifeTime = getOperandLifeTime(operation.inputs[2]);

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

    mPorts[operation.outputs[0]] = mBuilderModel->createFC(params, input);

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
    //[23]{30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52}
    IRBuilder::LstmLayer::LstmParams  params;

    // for each parameter we need to determine whether it is model input or not
    auto getOperandLifeTime = [&](uint32_t idx) {
        const auto op = mModel.operands[idx];
        return (int)op.lifetime;
    };

    auto getIRBlobFromOperand = [&](uint32_t idx, uint32_t offset) {
        const auto op = mModel.operands[idx];

        auto blob = GetConstOperandAsTensor(idx, offset);
        if (op.lifetime == OperandLifeTime::MODEL_INPUT)
        {
            mOpIndex2BlobMap[idx] = blob;
            VLOG(L1, "blob idx=%d (model_input) ptr=%p", idx, blob.get());
        }

        return blob;
    };

    // for (auto i=0; i < operation.inputs.size(); i++) {
    //      VLOG(L1, "input index = %d lifetime = %d", operation.inputs[i],
    //                                                 getOperandLifeTime(i));
    // }

    // for (auto i=0; i < operation.outputs.size(); i++) {
    //      VLOG(L1, "input index = %d lifetime = %d", operation.outputs[i],
    //                                                 getOperandLifeTime(i));
    // }

    IRBuilder::LstmLayer::LstmCellDescription lstmDesc;

    lstmDesc.clippingThresholdCellState     = PARAM_FP(21);
    lstmDesc.clippingThresholdProjState     = PARAM_FP(22);
    lstmDesc.cifgEnabled                    = false;
    lstmDesc.projectionLayerEnabled         = false;
    lstmDesc.peepholeEnabled                = false;

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

    if (operation.inputs.size() > 23) {
        if (!isOperandDataNull(operation.inputs[24]) &&
            !isOperandDataNull(operation.inputs[25]) &&
            !isOperandDataNull(operation.inputs[26])) {
            VLOG(L1, "Normalization weights are present.. Not supported yet.");
            return false;
        }
    }
    VLOG(L1, "Lstm cell description %s", lstmDescription.c_str());

    auto input          = getIRBlobFromOperand(operation.inputs[0], 0);
    auto outputStateIn  = getIRBlobFromOperand(operation.inputs[18], 18);
    auto cellStateIn    = getIRBlobFromOperand(operation.inputs[19], 19);

    VLOG(L1, "check point 2");

    params.input2inputWeights.data     = getIRBlobFromOperand(operation.inputs[1], 1);
    params.input2inputWeights.lifeTime = getOperandLifeTime(operation.inputs[1]);

    params.input2ForgetWeights.data     = getIRBlobFromOperand(operation.inputs[2], 2);
    params.input2ForgetWeights.lifeTime    = getOperandLifeTime(operation.inputs[2]);

    params.input2CellWeights.data     = getIRBlobFromOperand(operation.inputs[3], 3);
    params.input2CellWeights.lifeTime     = getOperandLifeTime(operation.inputs[3]);

    params.input2OutputWeights.data     = getIRBlobFromOperand(operation.inputs[4], 4);
    params.input2OutputWeights.lifeTime     = getOperandLifeTime(operation.inputs[4]);

    params.recurrant2inputWeights.data     = getIRBlobFromOperand(operation.inputs[5], 5);
    params.recurrant2inputWeights.lifeTime     = getOperandLifeTime(operation.inputs[5]);

    params.recurrant2ForgetWeights.data     = getIRBlobFromOperand(operation.inputs[6], 6);
    params.recurrant2ForgetWeights.lifeTime     = getOperandLifeTime(operation.inputs[6]);

    params.recurrant2CellWeights.data     = getIRBlobFromOperand(operation.inputs[7], 7);
    params.recurrant2CellWeights.lifeTime     = getOperandLifeTime(operation.inputs[7]);

    params.recurrant2OutputWeights.data     = getIRBlobFromOperand(operation.inputs[8], 8);
    params.recurrant2OutputWeights.lifeTime     = getOperandLifeTime(operation.inputs[8]);

    params.cell2InputWeights.data     = getIRBlobFromOperand(operation.inputs[9], 9);
    params.cell2InputWeights.lifeTime     = getOperandLifeTime(operation.inputs[9]);

    params.cell2ForgetWeights.data     = getIRBlobFromOperand(operation.inputs[10], 10);
    params.cell2ForgetWeights.lifeTime     = getOperandLifeTime(operation.inputs[10]);

    params.cell2OutputWeights.data  = getIRBlobFromOperand(operation.inputs[11], 11);
    params.cell2OutputWeights.lifeTime  = getOperandLifeTime(operation.inputs[11]);

    params.inputGateBias.data = getIRBlobFromOperand(operation.inputs[12], 12);
    params.inputGateBias.lifeTime = getOperandLifeTime(operation.inputs[12]);

    params.forgetGateBias.data   = getIRBlobFromOperand(operation.inputs[13], 13);
    params.forgetGateBias.lifeTime   = getOperandLifeTime(operation.inputs[13]);

    params.cellBias.data = getIRBlobFromOperand(operation.inputs[14], 14);
    params.cellBias.lifeTime = getOperandLifeTime(operation.inputs[14]);

    params.outputGateBias.data    = getIRBlobFromOperand(operation.inputs[15], 15);
    params.outputGateBias.lifeTime    = getOperandLifeTime(operation.inputs[15]);

    if (lstmDesc.projectionLayerEnabled) {
        params.projectionWeights.data       = getIRBlobFromOperand(operation.inputs[16], 16);
        params.projectionWeights.lifeTime       = getOperandLifeTime(operation.inputs[16]);

        params.projectionBias.data    = getIRBlobFromOperand(operation.inputs[17], 17);
        params.projectionBias.lifeTime    = getOperandLifeTime(operation.inputs[17]);
    }

    if (operation.inputs.size() > 23) {
        params.inputLayerNormWeights.data      = GetConstOperandAsTensor(operation.inputs[23], 23);
        params.inputLayerNormWeights.lifeTime = getOperandLifeTime(operation.inputs[23]);

        params.forgetLayerNormWeights.data     = GetConstOperandAsTensor(operation.inputs[24], 24);
        params.forgetLayerNormWeights.lifeTime = getOperandLifeTime(operation.inputs[24]);

        params.cellLayerNormWeights.data       = GetConstOperandAsTensor(operation.inputs[25], 25);
        params.cellLayerNormWeights.lifeTime = getOperandLifeTime(operation.inputs[25]);

        params.outputLayerNormWeights.data     = GetConstOperandAsTensor(operation.inputs[26], 26);
        params.outputLayerNormWeights.lifeTime = getOperandLifeTime(operation.inputs[26]);
    }

    params.activationFunction               = PARAM_I32(20);

    mPorts[operation.outputs[3]] = mBuilderModel->createFullLstm(params, lstmDesc, input, cellStateIn, outputStateIn);

    return true;
}

IRBlob::Ptr GnaPreparedModel::GetConstWeightsOperandAsTensor(uint32_t index)
{
    dumpOperand(index);
    const auto op = mModel.operands[index];
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
            InferenceEngine::TBlob<float>::Ptr blob =
                                std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
            blob->allocate();
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
    const auto op = mModel.operands[operand_index];
    uint32_t len;
    const uint8_t *buf = GetOperandMemory(mModel, operand_index, len);

    VLOG(L1, "GnaPreparedModel:: Operand: index: %d, len: %d, buf: %p", operand_index, len, buf);
    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
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
                InferenceEngine::TBlob<float>::Ptr blob =
                            std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
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

Blob::Ptr GnaPreparedModel::GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len)
{
    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        if (op.lifetime == OperandLifeTime::MODEL_INPUT) {
            if (buf == nullptr)
                VLOG(L1, "MODEL_INPUT buf is NULL !!!!!!!!!!!!!!!");

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
                //VLOG(L1, "buf data %f", *((float*)buf));
                //VLOG(L1, "buf data %f", *((float*)buf + 1));
                InferenceEngine::TBlob<float>::Ptr blob =
                                std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
                blob->allocate();
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
        } else if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
            if (buf == nullptr)
                VLOG(L1, "MODEL_OUTPUT buf is NULL !!!!!!!!!!!!!!!");

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
