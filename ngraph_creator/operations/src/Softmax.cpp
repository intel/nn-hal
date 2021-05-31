//#define LOG_NDEBUG 0
#include <Softmax.hpp>
#define LOG_TAG "Softmax"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Softmax::Softmax(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Softmax::validate() {
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }
    if (!checkInputOperandType(1, (int32_t)OperandType::FLOAT32)) {
        return false;
    }
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    if (inputsSize == 3) {
        if (!checkInputOperandType(2, (int32_t)OperandType::INT32)) {
            return false;
        }
    }
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Softmax::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        input = getInputNode<float>(0);
    } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        input = getInputNode<uint8_t>(0);
        const auto& inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
        input = DequantizeNode(input, inputIndex, ngraph::element::f32);
    }

    float beta = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 1);
    auto betaNode = ngraph::opset3::Constant::create(ngraph::element::f32, {}, {beta});
    int axis = -1;
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    if (inputsSize == 3) axis = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 2);

    const auto axisNode =
        ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {axis});
    // max(input[batch, :]
    auto max = std::make_shared<ngraph::opset3::ReduceMax>(input, axisNode, true);
    // input[batch, i] - max(input[batch, :])
    auto sub = std::make_shared<ngraph::opset3::Subtract>(input, max);
    // (input[batch, i] - max(input[batch, :])) * beta
    auto mul = std::make_shared<ngraph::opset3::Multiply>(sub, betaNode);
    // exp((input[batch, i] - max(input[batch, :])) * beta)
    auto exp = std::make_shared<ngraph::opset3::Exp>(mul);
    // sum_{k}{exp((input[batch, k] - max(input[batch, :])) * beta)}
    auto sum = std::make_shared<ngraph::opset3::ReduceSum>(exp, axisNode, true);

    std::shared_ptr<ngraph::Node> outputNode;
    // exp((input[batch, i] - max(input[batch, :])) * beta) / sum_{k}{exp((input[batch, k] -
    // max(input[batch, :])) * beta)}
    outputNode = std::make_shared<ngraph::opset3::Divide>(exp, sum);

    if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        const auto& outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
        outputNode = QuantizeNode(outputNode, outputIndex, ngraph::element::u8);
    }

    const auto outputOperand = sModelInfo->getOperand(mDefaultOutputIndex);
    if (outputOperand.lifetime == OperandLifeTime::MODEL_OUTPUT)
        addResultNode(mDefaultOutputIndex, outputNode);

    return outputNode;
}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
