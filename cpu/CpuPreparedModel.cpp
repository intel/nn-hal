#define LOG_TAG "CpuPreparedModel"

#include "CpuPreparedModel.h"
#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>
#include <fstream>
#include <thread>
#include "ValidateHal.h"
#include "utils.h"

using namespace android::nn;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

void CpuPreparedModel::deinitialize() {
    ALOGV("Entering %s", __func__);
    delete mEnginePtr;
    mEnginePtr = nullptr;
    ALOGI("free engine");
    ALOGV("Exiting %s", __func__);
}

bool CpuPreparedModel::initialize(const Model& model) {
    ALOGV("Entering %s", __func__);
    bool success = false;

    // NgraphNetworkCreator ngc(mModel, "CPU");
    if (!mNgc->validateOperations()) return false;
    mNgc->initializeModel();  // NgraphNetworkCreator
    auto ngraph_function = mNgc->generateGraph();
    InferenceEngine::CNNNetwork ngraph_net = InferenceEngine::CNNNetwork(ngraph_function);
    ngraph_net.serialize("/data/vendor/neuralnetworks/ngraph_ir.xml",
                         "/data/vendor/neuralnetworks/ngraph_ir.bin");

    // Check operation supoorted or not, user may not call getOpertionSupported()
    for (const auto& operation : mModel.operations) {
        success = isOperationSupported(operation, mModel);
        dumpOperationSupport(operation, success);
        if (!success) {
            ALOGE("Unsupported operation in initialize()");
            return false;
        }
    }

    success = setRunTimePoolInfosFromHidlMemories(&mPoolInfos, mModel.pools);
    if (!success) {
        ALOGE("setRunTimePoolInfosFromHidlMemories failed.");
        return false;
    }

    success = initializeRunTimeOperandInfo();
    if (!success) {
        ALOGE("initializeRunTimeOperandInfo failed.");
        return false;
    }

    ALOGI("initialize ExecuteNetwork for device %s", mTargetDevice.c_str());
    mEnginePtr = new ExecuteNetwork(ngraph_net, mTargetDevice);
    mEnginePtr->prepareInput();
    mEnginePtr->loadNetwork(ngraph_net);
    ALOGV("Exiting %s", __func__);
    return true;
}

IRBlob::Ptr CpuPreparedModel::GetConstWeightsOperandAsTensor(uint32_t index, const Model& model) {
    ALOGV("Entering %s", __func__);
    dumpOperand(index, model);
    const auto op = model.operands[index];
    uint32_t len;
    const uint8_t* buf = GetOperandMemory(model, index, len);
    ALOGI("CpuPreparedModel:: Operand: index: %d, len: %d, buf: %p", index, len, buf);
    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        vec<unsigned int> order;
        Layout layout;
        if (op.dimensions.size() == 4) {
            // order = {0,3,1,2};  //nhwc -> nchw
            order = {3, 0, 1, 2};   // IHWO -> OIHW for depth conv
            layout = Layout::OIHW;  // weights layout
        } else if (op.dimensions.size() == 2) {
            order = {0, 1};
            layout = Layout::NC;
        } else {
            order = {0};  //(op.dimensions.size() < 2)
            layout = Layout::C;
        }
        auto inputDims = toDims(op.dimensions);
        TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(inputDims, order), layout);
        if (buf == nullptr) {
            ALOGE("TENSOR_FLOAT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();
            return blob;
        } else {
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf, len);
            return blob;
        }
    } else if (op.type == OperandType::TENSOR_INT32) {
        ALOGD("check if const tensors of type IN32 supported");
        TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), Layout::ANY);
        if (buf == nullptr) {
            ALOGE("TENSOR_INT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();
            return blob;
        } else {
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf, len);
            return blob;
        }
    } else {
        ALOGE("not supporting const tensors of type ", op.type);
        nnAssert(false);
    }
    ALOGV("Exiting %s", __func__);
    return nullptr;
}

IRBlob::Ptr CpuPreparedModel::GetConstOperandAsTensor(int operand_idx, int operation_idx,
                                                      const Model& model) {
    ALOGV("Entering %s", __func__);
    dumpOperand(operand_idx, model);
    const auto op = model.operands[operand_idx];
    uint32_t len;

    const uint8_t* buf = GetOperandMemory(mModel, operand_idx, len);
    ALOGI("CpuPreparedModel:: operand_index: %d, operation_index :%d,len: %d, buf: %p", operand_idx,
          operation_idx, len, buf);

    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        vec<unsigned int> order;
        Layout layout;
        if (op.dimensions.size() == 4) {
            order = {0, 3, 1, 2};   // nhwc -> nchw
            layout = Layout::OIHW;  // weights layout
        } else if (op.dimensions.size() == 2) {
            order = {0, 1};
            layout = Layout::NC;
        } else {
            order = {0};  //(op.dimensions.size() < 2)
            layout = Layout::C;
        }
        auto inputDims = toDims(op.dimensions);
        TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(inputDims, order), layout);
        if (buf == nullptr) {
            ALOGE("TENSOR_FLOAT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();
            return blob;
        } else {
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf, len);
            return blob;
        }
    } else if (op.type == OperandType::TENSOR_INT32) {
        ALOGD("check if const tensors of type IN32 supported");
        TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), Layout::ANY);
        if (buf == nullptr) {
            ALOGE("TENSOR_INT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();
            return blob;
        } else {
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf, len);
            return blob;
        }
    } else {
        ALOGE("not supporting const tensors of type ", op.type);
        nnAssert(false);
    }
    ALOGV("Exiting %s", __func__);
    return nullptr;
}

Blob::Ptr CpuPreparedModel::GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t* buf,
                                                  uint32_t& len) {
    ALOGV("Entering %s", __func__);
    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        if (op.lifetime == OperandLifeTime::MODEL_INPUT) {
            ALOGI("Create input blob !!!!");
            std::vector<uint32_t> dims(op.dimensions.begin(), op.dimensions.end());
            vec<unsigned int> order;
            Layout layout;
            if (op.dimensions.size() == 4) {
                order = {0, 3, 1, 2};  // nhwc -> nchw
                layout = Layout::NCHW;
            } else if (op.dimensions.size() == 3) {  // Inputs are forced to 4D
                dims.insert(dims.begin(), 1);
                order = {0, 3, 1, 2};  // nhwc -> nchw
                layout = Layout::NCHW;
            } else if (op.dimensions.size() == 2) {
                order = {0, 1};
                layout = Layout::NC;
            } else {
                order = {0};  //(op.dimensions.size() < 2)
                layout = Layout::C;
            }
            ALOGD("GetInOutOperandAsBlob dims size %d", op.dimensions.size());
            auto inputDims = toDims(dims);
            TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(inputDims, order), layout);

            if (buf == nullptr) {
                ALOGE("MODEL_INPUT buf is NULL !!!!!!!!!!!!!!!");
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();
                return blob;
            } else {
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf, len);
                return blob;
            }
        } else if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
            ALOGI("Create output blob !!!!");
            vec<unsigned int> order;
            Layout layout;
            if (op.dimensions.size() == 4) {
                // order = {0,3,1,2};  //nhwc -> nchw
                layout = Layout::NHWC;
            } else if (op.dimensions.size() == 2) {
                // order = {0, 1};
                layout = Layout::NC;
            } else if (op.dimensions.size() == 3) {
                // order = {0, 1, 2, 3};  // nhwc -> nchw
                layout = Layout::CHW;
                ALOGD("Anoob : GetInOutOperandAsBlob output already transposed to NHWC");
            } else {
                // order = {0}; //(op.dimensions.size() < 2)
                layout = Layout::C;
            }

            TensorDesc td(InferenceEngine::Precision::FP32, toDims(op.dimensions), layout);  // nhwc
            if (buf == nullptr) {
                ALOGE("MODEL_OUTPUT buf is NULL !!!!!!!!!!!!!!!");
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();
                return blob;
            } else {
                InferenceEngine::TBlob<float>::Ptr blob =
                    InferenceEngine::make_shared_blob<float>(td, (float*)buf, len);
                return blob;
            }
        }
    } else if (op.type == OperandType::TENSOR_INT32) {
        ALOGD("check if const tensors of type IN32 supported");
        // nnAssert(true);
        TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), Layout::ANY);
        return std::make_shared<InferenceEngine::TBlob<int32_t>>(td, (int32_t*)buf, len);
    } else {
        ALOGE("not supporting const tensors of type ", op.type);
        nnAssert(false);
    }
    ALOGV("Exiting %s", __func__);
    return nullptr;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
