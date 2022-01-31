#include <OperationsFactory.hpp>
#undef LOG_TAG
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
        case OperationType::ABS:
            return std::make_shared<Abs>(operationIndex);
        case OperationType::ADD:
            return std::make_shared<Add>(operationIndex);
        case OperationType::ARGMAX:
            return std::make_shared<Argmax>(operationIndex);
        case OperationType::ARGMIN:
            return std::make_shared<Argmin>(operationIndex);
        case OperationType::AVERAGE_POOL_2D:
            return std::make_shared<AveragePool2D>(operationIndex);
        case OperationType::BATCH_TO_SPACE_ND:
            return std::make_shared<BatchToSpace>(operationIndex);
        case OperationType::BIDIRECTIONAL_SEQUENCE_RNN:
            return std::make_shared<BidirectionalSequenceRNN>(operationIndex);
        case OperationType::CAST:
            return std::make_shared<Cast>(operationIndex);
        case OperationType::CHANNEL_SHUFFLE:
            return std::make_shared<ChannelShuffle>(operationIndex);
        case OperationType::CONCATENATION:
            return std::make_shared<Concat>(operationIndex);
        case OperationType::CONV_2D:
            return std::make_shared<Conv2d>(operationIndex);
        case OperationType::DEPTH_TO_SPACE:
            return std::make_shared<DepthToSpace>(operationIndex);
        case OperationType::DEPTHWISE_CONV_2D:
            return std::make_shared<DepthwiseConv2d>(operationIndex);
        case OperationType::DEQUANTIZE:
            return std::make_shared<Dequantize>(operationIndex);
        case OperationType::DIV:
            return std::make_shared<Div>(operationIndex);
        case OperationType::EMBEDDING_LOOKUP:
            return std::make_shared<EmbeddingLookup>(operationIndex);
        case OperationType::EQUAL:
            return std::make_shared<Equal>(operationIndex);
        case OperationType::EXP:
            return std::make_shared<Exp>(operationIndex);
        case OperationType::EXPAND_DIMS:
            return std::make_shared<ExpandDims>(operationIndex);
        case OperationType::FULLY_CONNECTED:
            return std::make_shared<FullyConnected>(operationIndex);
        case OperationType::FLOOR:
            return std::make_shared<Floor>(operationIndex);
        case OperationType::GATHER:
            return std::make_shared<Gather>(operationIndex);
        case OperationType::GREATER:
            return std::make_shared<Greater>(operationIndex);
        case OperationType::GREATER_EQUAL:
            return std::make_shared<GreaterEqual>(operationIndex);
        case OperationType::GROUPED_CONV_2D:
            return std::make_shared<GroupedConv2d>(operationIndex);
        case OperationType::HARD_SWISH:
            return std::make_shared<HardSwish>(operationIndex);
        case OperationType::INSTANCE_NORMALIZATION:
            return std::make_shared<InstanceNormalization>(operationIndex);
        case OperationType::L2_POOL_2D:
            return std::make_shared<L2Pooling2D>(operationIndex);
        case OperationType::L2_NORMALIZATION:
            return std::make_shared<L2Normalization>(operationIndex);
        case OperationType::LSTM:
            return std::make_shared<LSTM>(operationIndex);
        case OperationType::LESS:
            return std::make_shared<Less>(operationIndex);
        case OperationType::LESS_EQUAL:
            return std::make_shared<LessEqual>(operationIndex);
        case OperationType::LOG_SOFTMAX:
            return std::make_shared<LogSoftmax>(operationIndex);
        case OperationType::LOG:
            return std::make_shared<Log>(operationIndex);
        case OperationType::LOGICAL_AND:
            return std::make_shared<LogicalAnd>(operationIndex);
        case OperationType::LOGICAL_NOT:
            return std::make_shared<LogicalNot>(operationIndex);
        case OperationType::LOGICAL_OR:
            return std::make_shared<LogicalOr>(operationIndex);
        case OperationType::LOGISTIC:
            return std::make_shared<Logistic>(operationIndex);
        case OperationType::MAXIMUM:
            return std::make_shared<Maximum>(operationIndex);
        case OperationType::MAX_POOL_2D:
            return std::make_shared<MaxPool2d>(operationIndex);
        case OperationType::MEAN:
            return std::make_shared<Mean>(operationIndex);
        case OperationType::MINIMUM:
            return std::make_shared<Minimum>(operationIndex);
        case OperationType::MUL:
            return std::make_shared<Mul>(operationIndex);
        case OperationType::NEG:
            return std::make_shared<Neg>(operationIndex);
        case OperationType::NOT_EQUAL:
            return std::make_shared<NotEqual>(operationIndex);
        case OperationType::PAD:
            return std::make_shared<Pad>(operationIndex);
        case OperationType::PAD_V2:
            return std::make_shared<PadV2>(operationIndex);
        case OperationType::POW:
            return std::make_shared<Pow>(operationIndex);
        case OperationType::PRELU:
            return std::make_shared<PRelu>(operationIndex);
        case OperationType::QUANTIZE:
            return std::make_shared<Quantize>(operationIndex);
        case OperationType::REDUCE_ALL:
            return std::make_shared<ReduceAll>(operationIndex);
        case OperationType::REDUCE_ANY:
            return std::make_shared<ReduceAny>(operationIndex);
        case OperationType::REDUCE_MAX:
            return std::make_shared<ReduceMax>(operationIndex);
        case OperationType::REDUCE_MIN:
            return std::make_shared<ReduceMin>(operationIndex);
        case OperationType::REDUCE_PROD:
            return std::make_shared<ReduceProd>(operationIndex);
        case OperationType::REDUCE_SUM:
            return std::make_shared<ReduceSum>(operationIndex);
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
        case OperationType::ROI_ALIGN:
            return std::make_shared<ROIAlign>(operationIndex);
        case OperationType::ROI_POOLING:
            return std::make_shared<ROIPooling>(operationIndex);
        case OperationType::RSQRT:
            return std::make_shared<RSQRT>(operationIndex);
        case OperationType::RESIZE_BILINEAR:
            return std::make_shared<ResizeBilinear>(operationIndex);
        case OperationType::RESIZE_NEAREST_NEIGHBOR:
            return std::make_shared<ResizeNearestNeighbor>(operationIndex);
        case OperationType::SELECT:
            return std::make_shared<Select>(operationIndex);
        case OperationType::SOFTMAX:
            return std::make_shared<Softmax>(operationIndex);
        case OperationType::SPACE_TO_BATCH_ND:
            return std::make_shared<SpaceToBatch>(operationIndex);
        case OperationType::SPACE_TO_DEPTH:
            return std::make_shared<SpaceToDepth>(operationIndex);
        case OperationType::SQRT:
            return std::make_shared<SQRT>(operationIndex);
        case OperationType::SIN:
            return std::make_shared<Sin>(operationIndex);
        case OperationType::SPLIT:
            return std::make_shared<Split>(operationIndex);
        case OperationType::STRIDED_SLICE:
            return std::make_shared<StridedSlice>(operationIndex);
        case OperationType::SQUEEZE:
            return std::make_shared<Squeeze>(operationIndex);
        case OperationType::SUB:
            return std::make_shared<Sub>(operationIndex);
        case OperationType::TANH:
            return std::make_shared<Tanh>(operationIndex);
        case OperationType::TOPK_V2:
            return std::make_shared<TopkV2>(operationIndex);
        case OperationType::TRANSPOSE_CONV_2D:
            return std::make_shared<TransposeConv2D>(operationIndex);
        case OperationType::TRANSPOSE:
            return std::make_shared<Transpose>(operationIndex);
        case OperationType::UNIDIRECTIONAL_SEQUENCE_RNN:
            return std::make_shared<UnidirectionalSequenceRNN>(operationIndex);
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
