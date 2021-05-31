#include <Quantize.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Quantize::Quantize(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Quantize::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Quantize::createNode() {
    // Creating input nodes
    auto input = getInputNode<float>(0);

    auto inputElementType = input->get_element_type();

    const auto& outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
    float outputScale = sModelInfo->getOperandScale(outputIndex);
    int outputZeroPoint = sModelInfo->getOperandZeroPoint(outputIndex);

    auto scale = ngraph::op::Constant::create(inputElementType, ngraph::Shape{}, {outputScale});
    auto zeroPoint =
        ngraph::op::Constant::create(ngraph::element::i32, ngraph::Shape{}, {outputZeroPoint});
    auto minVal = ngraph::op::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0});
    auto maxVal = ngraph::op::Constant::create(ngraph::element::i32, ngraph::Shape{}, {255});

    auto div = std::make_shared<ngraph::opset3::Divide>(input, scale);
    ngraph::op::v5::Round::RoundMode mode = ngraph::op::v5::Round::RoundMode::HALF_TO_EVEN;
    auto round = std::make_shared<ngraph::op::v5::Round>(div, mode);
    auto convertRound = std::make_shared<ngraph::opset3::Convert>(round, ngraph::element::i32);
    auto sum = std::make_shared<ngraph::opset3::Add>(convertRound, zeroPoint);
    auto min = std::make_shared<ngraph::opset3::Minimum>(maxVal, sum);
    auto max = std::make_shared<ngraph::opset3::Maximum>(minVal, min);

    ngraph::element::Type elementType = ngraph::element::u8;

    auto outputNode = std::make_shared<ngraph::opset3::Convert>(max, elementType);

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
