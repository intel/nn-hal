#include <Pad.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Pad::Pad(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Pad::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }
    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_INT32)) {
        return false;
    }

    // Check input rank
    const auto inputRank = getInputOperandDimensions(0).size();
    if (inputRank > 4) return false;

    // TODO: Add support for low_rank
    if (inputRank < 2) return false;

    // TODO: Add Support for all_tensors_as_inputs
    const auto& padOperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);

    if (!sModelInfo->isOperandLifeTimeConst(padOperandIndex)) {
        ALOGE("%s Only Constant dimensions supported now", __func__);
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Pad::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> inputNode;
    auto paddings = getInputNode(1);

    inputNode = getInputNode(0);

    const auto axisNode = createConstNode(ngraph::element::i32, {}, convertToVector(1));
    auto paddingsSplitNode =
        std::make_shared<ngraph::opset3::Split>(paddings, axisNode, 2)->outputs();

    const auto shapeNode = createConstNode(
        ngraph::element::i32, {1}, convertToVector((int32_t)getInputOperandDimensions(0).size()));

    std::shared_ptr<ngraph::Node> pads_begin =
        std::make_shared<ngraph::opset3::Reshape>(paddingsSplitNode[0], shapeNode, true);

    std::shared_ptr<ngraph::Node> pads_end =
        std::make_shared<ngraph::opset3::Reshape>(paddingsSplitNode[1], shapeNode, true);

    auto outputNode = std::make_shared<ngraph::opset3::Pad>(inputNode, pads_begin, pads_end,
                                                            ngraph::op::PadMode::CONSTANT);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
