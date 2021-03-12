#include <Reshape.hpp>
#define LOG_TAG "Reshape"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Reshape::Reshape(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Reshape::validate() { return true; }

std::shared_ptr<ngraph::Node> Reshape::createNode() {
    const auto& dimsOperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
    auto outDims = sModelInfo->GetConstVecOperand<int32_t>(dimsOperandIndex);
    VLOGDIMS(L3, outDims, "Reshape::createNode dims");
    auto inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
    auto inputOp = mNgraphNodes->getOperationOutput(inputIndex);
    ALOGV("%s outDims.size=%d", __func__, outDims.size());

    if (mNgraphNodes->isForcedNchw(inputIndex)) {
        inputOp = transpose(NCHW_NHWC, inputOp);
        mNgraphNodes->setForcedNchw(mDefaultOutputIndex, false);
    }

    auto shapeNode = std::make_shared<ngraph::opset3::Constant>(
        ngraph::element::i32, ngraph::Shape{outDims.size()}, outDims.data());

    std::shared_ptr<ngraph::Node> outputNode =
        std::make_shared<ngraph::opset3::Reshape>(inputOp, shapeNode, true);

    const auto outputOperand = sModelInfo->getOperand(mDefaultOutputIndex);
    if (outputOperand.lifetime == OperandLifeTime::MODEL_OUTPUT)
        addResultNode(mDefaultOutputIndex, outputNode);

    return outputNode;
}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
