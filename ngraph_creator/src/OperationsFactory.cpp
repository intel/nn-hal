#include <OperationsFactory.hpp>
#define LOG_TAG "OperationsFactory"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

OperationsFactory::OperationsFactory(const std::string& plugin,
                                     std::shared_ptr<NnapiModelInfo> modelInfo,
                                     std::shared_ptr<NgraphNodes> nodes) {
    OperationsBase::sPluginType = plugin;
    OperationsBase::sModelInfo = modelInfo;
    OperationsBase::mNgraphNodes = nodes;
    ALOGV("%s Constructed", __func__);
}
OperationsFactory::~OperationsFactory() {
    OperationsBase::mNgraphNodes.reset();
    ALOGV("%s Destructed & reset", __func__);
}
std::shared_ptr<OperationsBase> OperationsFactory::getOperation(
    int operationIndex, const OperationType& operationType) {
    switch (operationType) {
        case OperationType::ABS:
            return std::make_shared<Abs>(operationIndex);
        case OperationType::ADD:
            return std::make_shared<Add>(operationIndex);
        case OperationType::CAST:
            return std::make_shared<Cast>(operationIndex);
        case OperationType::CONCATENATION:
            return std::make_shared<Concat>(operationIndex);
        case OperationType::CONV_2D:
            return std::make_shared<Conv_2d>(operationIndex);
        case OperationType::DEPTHWISE_CONV_2D:
            return std::make_shared<Depthwise_Conv_2d>(operationIndex);
        case OperationType::DIV:
            return std::make_shared<Div>(operationIndex);
        case OperationType::EQUAL:
            return std::make_shared<Equal>(operationIndex);
        case OperationType::EXP:
            return std::make_shared<Exp>(operationIndex);
        case OperationType::FLOOR:
            return std::make_shared<Floor>(operationIndex);
        case OperationType::GATHER:
            return std::make_shared<Gather>(operationIndex);
        case OperationType::GREATER:
            return std::make_shared<Greater>(operationIndex);
        case OperationType::GREATER_EQUAL:
            return std::make_shared<Greater_Equal>(operationIndex);
        case OperationType::LSTM:
            return std::make_shared<LSTM>(operationIndex);
        case OperationType::LESS:
            return std::make_shared<Less>(operationIndex);
        case OperationType::LESS_EQUAL:
            return std::make_shared<Less_Equal>(operationIndex);
        case OperationType::LOG:
            return std::make_shared<Log>(operationIndex);
        case OperationType::RESHAPE:
            return std::make_shared<Reshape>(operationIndex);
        case OperationType::SOFTMAX:
            return std::make_shared<Softmax>(operationIndex);
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
