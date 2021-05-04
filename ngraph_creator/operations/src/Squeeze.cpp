#include <Squeeze.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Squeeze::Squeeze(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Squeeze::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) return false;

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) return false;

    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_INT32)) return false;

    return true;
}

std::shared_ptr<ngraph::Node> Squeeze::createNode() {
    // Creating input nodes
    auto input = getInputNode<float>(0);

    std::shared_ptr<ngraph::Node> dims;

    if (!sModelInfo->isOmittedInput(mNnapiOperationIndex, 1))
        dims = getInputNode<int>(1);
    else
        dims = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});

    auto outputNode = std::make_shared<ngraph::opset3::Squeeze>(input, dims);

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
