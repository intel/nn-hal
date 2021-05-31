#include <Mul.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Mul::Mul(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Mul::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    // Check all input types
    for (int i = 0; i <= 1; i++) {
        if (!checkInputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32) &&
            !checkInputOperandType(i, (int32_t)OperandType::TENSOR_QUANT8_ASYMM))
            return false;
    }

    if (!checkInputOperandType(2, (int32_t)OperandType::INT32)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Mul::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input1, input2;
    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        input1 = getInputNode<float>(0);
        input2 = getInputNode<float>(1);
    } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        input1 = getInputNode<uint8_t>(0);
        input2 = getInputNode<uint8_t>(1);

        const auto& input1Index = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
        const auto& input2Index = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
        input1 = DequantizeNode(input1, input1Index, ngraph::element::f32);
        input2 = DequantizeNode(input2, input2Index, ngraph::element::f32);
    }

    auto activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);

    auto mulNode = std::make_shared<ngraph::opset3::Multiply>(input1, input2,
                                                              ngraph::op::AutoBroadcastType::NUMPY);

    auto outputNode = applyActivation(mulNode, activationFn);

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
