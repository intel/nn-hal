#include <Pad_V2.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Pad_V2::Pad_V2(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Pad_V2::validate() {
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
    if (!checkInputOperandType(2, (int32_t)OperandType::FLOAT32) &&
        !checkInputOperandType(2, (int32_t)OperandType::INT32))
        return false;

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

std::shared_ptr<ngraph::Node> Pad_V2::createNode() {
    // Creating input nodes
    auto inputNode = getInputNode(0);
    auto paddings = getInputNode(1);
    std::shared_ptr<ngraph::Node> pad_value;
    auto inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        auto pad_scalar_value = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 2);
        pad_value = createConstNode(ngraph::element::f32, {}, convertToVector(pad_scalar_value));
    } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        auto pad_scalar_value = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 2);
        pad_value = createConstNode(ngraph::element::i32, {}, convertToVector(pad_scalar_value));

        // scale and zeropoint of pad value has to be same as in inputNode. so inputIndex is passed
        // as second parameter to DequantizeNode
        pad_value = DequantizeNode(pad_value, inputIndex, ngraph::element::f32);
    }

    const auto axisNode = createConstNode(ngraph::element::i32, {}, convertToVector(1));
    auto paddingsSplitNode =
        std::make_shared<ngraph::opset3::Split>(paddings, axisNode, 2)->outputs();

    const auto shapeNode = createConstNode(
        ngraph::element::i32, {1}, convertToVector((int32_t)getInputOperandDimensions(0).size()));

    std::shared_ptr<ngraph::Node> pads_begin =
        std::make_shared<ngraph::opset3::Reshape>(paddingsSplitNode[0], shapeNode, true);

    std::shared_ptr<ngraph::Node> pads_end =
        std::make_shared<ngraph::opset3::Reshape>(paddingsSplitNode[1], shapeNode, true);

    auto outputNode = std::make_shared<ngraph::opset3::Pad>(
        inputNode, pads_begin, pads_end, pad_value, ngraph::op::PadMode::CONSTANT);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
