#include <Minimum.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Minimum::Minimum(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Minimum::validate() {
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
    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(1, (int32_t)OperandType::TENSOR_INT32) &&
        !checkInputOperandType(1, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Minimum::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input1, input2;

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        input1 = getInputNode<float>(0);
        input2 = getInputNode<float>(1);
    } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_INT32)) {
        input1 = getInputNode<int>(0);
        input2 = getInputNode<int>(1);
    } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        input1 = getInputNode<uint8_t>(0);
        input2 = getInputNode<uint8_t>(1);

        const auto& input1Index = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
        const auto& input2Index = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
        input1 = DequantizeNode(input1, input1Index, ngraph::element::f32);
        input2 = DequantizeNode(input2, input2Index, ngraph::element::f32);
    }

    std::shared_ptr<ngraph::Node> outputNode;

    outputNode = std::make_shared<ngraph::opset3::Minimum>(input1, input2,
                                                           ngraph::op::AutoBroadcastType::NUMPY);

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
