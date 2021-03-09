#include <Reshape.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Reshape::Reshape(const Model& model) : OperationsBase(model) {}

bool Reshape::validate(const Operation& op) { return true; }

std::shared_ptr<ngraph::Node> Reshape::createNode(const Operation& operation) {
    auto outDims = GetConstVecOperand(mModel, operation.inputs[1]);
    auto inputIndex = operation.inputs[0];
    auto inputOp = mNgraphNodes->getOperationOutput(inputIndex);
    const auto inputOperand = mModel.operands[inputIndex];
    const auto outputIndex = operation.outputs[0];

    if (mNgraphNodes->isForcedNchw(inputIndex)) {
        inputOp = transpose(NCHW_NHWC, inputOp);
        mNgraphNodes->setForcedNchw(outputIndex, false);
    }

    auto shapeNode = std::make_shared<ngraph::opset3::Constant>(
        ngraph::element::i64, ngraph::Shape{outDims.size()}, outDims.data());

    std::shared_ptr<ngraph::Node> outputNode =
        std::make_shared<ngraph::opset3::Reshape>(inputOp, shapeNode, true);

    const auto outputOperand = mModel.operands[outputIndex];
    if (outputOperand.lifetime == OperandLifeTime::MODEL_OUTPUT)
        addResultNode(outputIndex, outputNode);

    return outputNode;
}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
