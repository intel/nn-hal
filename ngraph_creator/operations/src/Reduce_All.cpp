#include <Reduce_All.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Reduce_All::Reduce_All(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Reduce_All::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_BOOL8)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_BOOL8)) return false;

    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_INT32)) return false;

    if (!checkInputOperandType(2, (int32_t)OperandType::BOOL)) return false;

    return true;
}

std::shared_ptr<ngraph::Node> Reduce_All::createNode() {
    // Creating input nodes
    auto input = getInputNode<bool>(0);
    auto reduction_axes = getInputNode<int>(1);
    auto keep_dims = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 2);

    auto outputNode =
        std::make_shared<ngraph::opset3::ReduceLogicalAnd>(input, reduction_axes, keep_dims);

    const auto op = sModelInfo->getOperand(mDefaultOutputIndex);
    if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
        addResultNode(mDefaultOutputIndex, outputNode);
    }
    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
