#include <Reshape.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Reshape::Reshape(NnapiModelInfo* model) : OperationsBase(model) {}

bool Reshape::validate(const Operation& op) { return true; }

std::shared_ptr<ngraph::Node> Reshape::createNode(const Operation& operation) {
    auto outDims = mModelInfo->GetConstVecOperand<uint32_t>(operation.inputs[1]);
    auto inputIndex = operation.inputs[0];
    auto inputOp = mNgraphNodes->getOperationOutput(inputIndex);
    const auto op = mModelInfo->getOperand(inputIndex);

    if (op.lifetime != OperandLifeTime::CONSTANT_COPY &&
        op.lifetime != OperandLifeTime::CONSTANT_REFERENCE &&
        op.lifetime != OperandLifeTime::MODEL_INPUT)
        inputOp = transpose(NCHW_NHWC, inputOp);

    if (outDims.size() == 3) outDims.insert(outDims.begin(), 1);

    auto shapeNode = std::make_shared<ngraph::opset3::Constant>(
        ngraph::element::i64, ngraph::Shape{outDims.size()}, outDims.data());

    auto reshape = std::make_shared<ngraph::opset3::Reshape>(inputOp, shapeNode, true);

    return transpose(NHWC_NCHW, reshape);
}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
