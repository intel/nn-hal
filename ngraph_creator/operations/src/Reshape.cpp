#include <Reshape.hpp>
#define LOG_TAG "Reshape"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Reshape::Reshape(const Operation& op) : OperationsBase(op) {
    mDefaultOutputIndex = mNnapiOp.outputs[0];
}

bool Reshape::validate() { return true; }

std::shared_ptr<ngraph::Node> Reshape::createNode() {
    auto outDims = GetConstVecOperand(*sModel, mNnapiOp.inputs[1]);
    auto inputIndex = mNnapiOp.inputs[0];
    auto inputOp = mNgraphNodes->getOperationOutput(inputIndex);
    const auto inputOperand = sModel->operands[inputIndex];
    ALOGV("%s outDims.size=%d", __func__, outDims.size());

    if (mNgraphNodes->isForcedNchw(inputIndex)) {
        inputOp = transpose(NCHW_NHWC, inputOp);
        mNgraphNodes->setForcedNchw(mDefaultOutputIndex, false);
    }

    auto shapeNode = std::make_shared<ngraph::opset3::Constant>(
        ngraph::element::i64, ngraph::Shape{outDims.size()}, outDims.data());

    std::shared_ptr<ngraph::Node> outputNode =
        std::make_shared<ngraph::opset3::Reshape>(inputOp, shapeNode, true);

    const auto outputOperand = sModel->operands[mDefaultOutputIndex];
    if (outputOperand.lifetime == OperandLifeTime::MODEL_OUTPUT)
        addResultNode(mDefaultOutputIndex, outputNode);

    return outputNode;
}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
