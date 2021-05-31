#include <Dequantize.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Dequantize::Dequantize(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Dequantize::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_SYMM)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Dequantize::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM))
        input = getInputNode<uint8_t>(0);
    else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_SYMM))
        input = getInputNode<int8_t>(0);

    auto elementType = ngraph::element::f32;

    const auto& inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
    float inputScale = sModelInfo->getOperandScale(inputIndex);
    int inputZeroPoint = sModelInfo->getOperandZeroPoint(inputIndex);

    auto scale = ngraph::op::Constant::create(elementType, ngraph::Shape{}, {inputScale});
    auto zeroPoint =
        ngraph::op::Constant::create(ngraph::element::i32, ngraph::Shape{}, {inputZeroPoint});

    auto convertInput = std::make_shared<ngraph::opset3::Convert>(input, ngraph::element::i32);
    auto diff = std::make_shared<ngraph::opset3::Subtract>(convertInput, zeroPoint);
    auto convertDiff = std::make_shared<ngraph::opset3::Convert>(diff, elementType);

    auto outputNode = std::make_shared<ngraph::opset3::Multiply>(convertDiff, scale);

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
