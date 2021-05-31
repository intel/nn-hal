#include <Cast.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Cast::Cast(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Cast::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_INT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_INT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Cast::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        input = getInputNode<float>(0);
    } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_INT32)) {
        input = getInputNode<int>(0);
    } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        input = getInputNode<uint8_t>(0);
    }

    ngraph::element::Type elementType;  // change to outputbased element type

    if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        elementType = ngraph::element::f32;
    } else if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_INT32)) {
        elementType = ngraph::element::i32;
    } else if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        const auto& outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
        auto minVal = ngraph::op::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0});
        auto maxVal = ngraph::op::Constant::create(ngraph::element::i32, ngraph::Shape{}, {255});
        auto convertInput = std::make_shared<ngraph::opset3::Convert>(input, ngraph::element::i32);
        auto min = std::make_shared<ngraph::opset3::Minimum>(maxVal, convertInput);
        input = std::make_shared<ngraph::opset3::Maximum>(minVal, min);
        elementType = ngraph::element::u8;
    }

    auto outputNode = std::make_shared<ngraph::opset3::Convert>(input, elementType);

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
