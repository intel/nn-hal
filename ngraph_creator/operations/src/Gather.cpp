#include <Gather.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Gather::Gather(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Gather::validate() {
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

    if (!checkInputOperandType(1, (int32_t)OperandType::INT32)) {
        return false;
    }

    if (!checkInputOperandType(2, (int32_t)OperandType::TENSOR_INT32)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Gather::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> gatherVals;

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        gatherVals = getInputNode<float>(0);
    } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_INT32)) {
        gatherVals = getInputNode<int>(0);
    } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        gatherVals = getInputNode<uint8_t>(0);

        const auto& inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
        gatherVals = DequantizeNode(gatherVals, inputIndex, ngraph::element::f32);
    }
    // axis range [-n, n]
    auto axis = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);
    auto axisNode = ngraph::opset3::Constant::create(ngraph::element::i64, {}, {axis});
    auto indices = getInputNode<int>(2);

    std::shared_ptr<ngraph::Node> outputNode;
    outputNode = std::make_shared<ngraph::opset3::Gather>(gatherVals, indices, axisNode);

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
