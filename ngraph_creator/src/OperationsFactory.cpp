#include <OperationsFactory.hpp>
#define LOG_TAG "OperationsFactory"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

OperationsFactory::OperationsFactory(IntelDeviceType deviceType,
                                     std::shared_ptr<NnapiModelInfo> modelInfo,
                                     std::shared_ptr<NgraphNodes> nodes) {
    OperationsBase::sPluginType = deviceType;
    OperationsBase::sModelInfo = modelInfo;
    ALOGV("%s Constructed", __func__);
}
OperationsFactory::~OperationsFactory() { ALOGV("%s Destructed", __func__); }
std::shared_ptr<OperationsBase> OperationsFactory::getOperation(
    int operationIndex, const OperationType& operationType) {
    switch (operationType) {
        case OperationType::ADD:
            return std::make_shared<Add>(operationIndex);
        case OperationType::AVERAGE_POOL_2D:
            return std::make_shared<Average_Pool_2D>(operationIndex);
        case OperationType::BATCH_TO_SPACE_ND:
            return std::make_shared<Batch_To_Space>(operationIndex);
        case OperationType::CONCATENATION:
            return std::make_shared<Concat>(operationIndex);
        case OperationType::CONV_2D:
            return std::make_shared<Conv_2d>(operationIndex);
        case OperationType::DEPTH_TO_SPACE:
            return std::make_shared<Depth_To_Space>(operationIndex);
        case OperationType::DEPTHWISE_CONV_2D:
            return std::make_shared<Depthwise_Conv_2d>(operationIndex);
        case OperationType::DEQUANTIZE:
            return std::make_shared<Dequantize>(operationIndex);
        case OperationType::DIV:
            return std::make_shared<Div>(operationIndex);
        case OperationType::FULLY_CONNECTED:
            return std::make_shared<FullyConnected>(operationIndex);
        case OperationType::FLOOR:
            return std::make_shared<Floor>(operationIndex);
        case OperationType::L2_POOL_2D:
            return std::make_shared<L2Pooling2D>(operationIndex);
        case OperationType::L2_NORMALIZATION:
            return std::make_shared<L2_Normalization>(operationIndex);
        case OperationType::LSTM:
            return std::make_shared<LSTM>(operationIndex);
        case OperationType::LOGISTIC:
            return std::make_shared<Logistic>(operationIndex);
        case OperationType::MAX_POOL_2D:
            return std::make_shared<Max_Pool_2d>(operationIndex);
        case OperationType::MEAN:
            return std::make_shared<Mean>(operationIndex);
        case OperationType::MUL:
            return std::make_shared<Mul>(operationIndex);
        case OperationType::PAD:
            return std::make_shared<Pad>(operationIndex);
        case OperationType::RELU:
            return std::make_shared<Relu>(operationIndex);
        case OperationType::RELU1:
            return std::make_shared<Relu1>(operationIndex);
        case OperationType::RELU6:
            return std::make_shared<Relu6>(operationIndex);
        case OperationType::RESHAPE:
            return std::make_shared<Reshape>(operationIndex);
        case OperationType::RNN:
            return std::make_shared<RNN>(operationIndex);
        case OperationType::RESIZE_BILINEAR:
            return std::make_shared<ResizeBilinear>(operationIndex);
        case OperationType::SOFTMAX:
            return std::make_shared<Softmax>(operationIndex);
        case OperationType::SPACE_TO_BATCH_ND:
            return std::make_shared<Space_To_Batch>(operationIndex);
        case OperationType::SPACE_TO_DEPTH:
            return std::make_shared<Space_To_Depth>(operationIndex);
        case OperationType::SQUEEZE:
            return std::make_shared<Squeeze>(operationIndex);
        case OperationType::SUB:
            return std::make_shared<Sub>(operationIndex);
        case OperationType::TANH:
            return std::make_shared<Tanh>(operationIndex);
        case OperationType::TRANSPOSE:
            return std::make_shared<Transpose>(operationIndex);
        default:
            ALOGE("%s Cannot identify OperationType %d", __func__, operationType);
            break;
    }
    return nullptr;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
