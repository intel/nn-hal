#include <Select.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Select::Select(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Select::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_INT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_BOOL8)) {
        return false;
    }

    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(1, (int32_t)OperandType::TENSOR_INT32) &&
        !checkInputOperandType(1, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    if (!checkInputOperandType(2, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(2, (int32_t)OperandType::TENSOR_INT32) &&
        !checkInputOperandType(2, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }
    return true;
}

std::shared_ptr<ngraph::Node> Select::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input1, input2, input3;
    input1 = getInputNode<uint8_t>(0);

    if (checkInputOperandType(1, (int32_t)OperandType::TENSOR_FLOAT32)) {
        input2 = getInputNode<float>(1);
        input3 = getInputNode<float>(2);
    } else if (checkInputOperandType(1, (int32_t)OperandType::TENSOR_INT32)) {
        input2 = getInputNode<int>(1);
        input3 = getInputNode<int>(2);
    } else if (checkInputOperandType(1, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        input2 = getInputNode<uint8_t>(1);
        input3 = getInputNode<uint8_t>(2);
        const auto& input2Index = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
        const auto& input3Index = sModelInfo->getOperationInput(mNnapiOperationIndex, 2);
        input2 = DequantizeNode(input2, input2Index, ngraph::element::f32);
        input3 = DequantizeNode(input3, input3Index, ngraph::element::f32);
    }

    std::shared_ptr<ngraph::Node> outputNode;

    outputNode = std::make_shared<ngraph::opset3::Select>(input1, input2, input3);

    if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        const auto& outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
        outputNode = QuantizeNode(outputNode, outputIndex, ngraph::element::u8);
    }

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
