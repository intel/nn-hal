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
    ALOGV("%s Constructed", __func__);
}
OperationsFactory::~OperationsFactory() { ALOGV("%s Destructed", __func__); }
std::shared_ptr<OperationsBase> OperationsFactory::getOperation(
    int operationIndex, const OperationType& operationType) {
    switch (operationType) {
        case OperationType::ABS:
            return std::make_shared<Abs>(operationIndex);
        case OperationType::ADD:
            return std::make_shared<Add>(operationIndex);
        case OperationType::AVERAGE_POOL_2D:
            return std::make_shared<Average_Pool_2D>(operationIndex);
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
        case OperationType::LOGICAL_AND:
            return std::make_shared<Logical_And>(operationIndex);
        case OperationType::LOGICAL_NOT:
            return std::make_shared<Logical_Not>(operationIndex);
        case OperationType::LOGICAL_OR:
            return std::make_shared<Logical_Or>(operationIndex);
        case OperationType::LOGISTIC:
            return std::make_shared<Logistic>(operationIndex);
        case OperationType::MAXIMUM:
            return std::make_shared<Maximum>(operationIndex);
        case OperationType::MAX_POOL_2D:
            return std::make_shared<Max_Pool_2d>(operationIndex);
        case OperationType::MEAN:
            return std::make_shared<Mean>(operationIndex);
        case OperationType::MINIMUM:
            return std::make_shared<Minimum>(operationIndex);
        case OperationType::MUL:
            return std::make_shared<Mul>(operationIndex);
        case OperationType::NEG:
            return std::make_shared<Neg>(operationIndex);
        case OperationType::NOT_EQUAL:
            return std::make_shared<Not_Equal>(operationIndex);
        case OperationType::POW:
            return std::make_shared<Pow>(operationIndex);
        case OperationType::REDUCE_ALL:
            return std::make_shared<Reduce_All>(operationIndex);
        case OperationType::REDUCE_ANY:
            return std::make_shared<Reduce_Any>(operationIndex);
        case OperationType::REDUCE_MIN:
            return std::make_shared<Reduce_Min>(operationIndex);
        case OperationType::REDUCE_PROD:
            return std::make_shared<Reduce_Prod>(operationIndex);
        case OperationType::RELU:
            return std::make_shared<Relu>(operationIndex);
        case OperationType::RELU1:
            return std::make_shared<Relu1>(operationIndex);
        case OperationType::RELU6:
            return std::make_shared<Relu6>(operationIndex);
        case OperationType::RESHAPE:
            return std::make_shared<Reshape>(operationIndex);
        case OperationType::SOFTMAX:
            return std::make_shared<Softmax>(operationIndex);
        case OperationType::SQRT:
            return std::make_shared<SQRT>(operationIndex);
        case OperationType::SIN:
            return std::make_shared<Sin>(operationIndex);
        case OperationType::SPLIT:
            return std::make_shared<Split>(operationIndex);
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
