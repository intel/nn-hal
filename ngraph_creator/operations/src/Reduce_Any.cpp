#include <Reduce_Any.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Reduce_Any::Reduce_Any(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Reduce_Any::validate() {
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

std::shared_ptr<ngraph::Node> Reduce_Any::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);
    auto reduction_axes = getInputNode(1);
    auto keep_dims = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 2);

    auto outputNode =
        std::make_shared<ngraph::opset3::ReduceLogicalOr>(input, reduction_axes, keep_dims);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
