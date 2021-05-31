#include <Squeeze.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Squeeze::Squeeze(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Squeeze::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM))
        return false;

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM))
        return false;

    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_INT32)) return false;

    // TODO: Add Support for all_tensors_as_inputs
    const auto& dimsOperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);

    // TODO: Support OmittedInput.
    // The empty 2nd argument in Squeeze op causes dynamic output
    // To add support, the dims will have to be calculated statically
    if (sModelInfo->isOmittedInput(mNnapiOperationIndex, 1) ||
        !sModelInfo->isOperandLifeTimeConst(dimsOperandIndex)) {
        ALOGE("%s Only Constant dimensions supported now", __func__);
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Squeeze::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        input = getInputNode<float>(0);
    } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        input = getInputNode<uint8_t>(0);
        const auto& inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
        input = DequantizeNode(input, inputIndex, ngraph::element::f32);
    }

    std::shared_ptr<ngraph::Node> dims;

    if (!sModelInfo->isOmittedInput(mNnapiOperationIndex, 1))
        dims = getInputNode<int>(1);
    else
        dims = make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{0},
                                                     std::vector<int64_t>{});

    std::shared_ptr<ngraph::Node> outputNode;

    outputNode = std::make_shared<ngraph::opset3::Squeeze>(input, dims);

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
