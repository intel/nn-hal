#include <Reduce_Min.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Reduce_Min::Reduce_Min(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Reduce_Min::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM))
        return false;

    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_INT32)) return false;

    if (!checkInputOperandType(2, (int32_t)OperandType::BOOL)) return false;

    return true;
}

std::shared_ptr<ngraph::Node> Reduce_Min::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;

    input = getInputNode(0);

    auto reduction_axes = getInputNode(1);
    auto keep_dims = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 2);

    std::shared_ptr<ngraph::Node> outputNode;
    outputNode = std::make_shared<ngraph::opset3::ReduceMin>(input, reduction_axes, keep_dims);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
