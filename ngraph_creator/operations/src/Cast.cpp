#include <Cast.hpp>
#undef LOG_TAG
#define LOG_TAG "Cast"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Cast::Cast(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

void Cast::connectOperationToGraph() { createNode(); }

std::shared_ptr<ngraph::Node> Cast::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;

    input = getInputNode(0, false);

    auto inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
    auto outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);

    const auto& inputType = sModelInfo->getOperationType(inputIndex);
    const auto& outputType = sModelInfo->getOperationType(outputIndex);

    ngraph::element::Type elementType;  // change to outputbased element type
    std::shared_ptr<ngraph::Node> outputNode;

    if (inputType == outputType) {
        outputNode = input;
    } else {
        if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
            elementType = ngraph::element::f32;
        } else if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT16)) {
            elementType = ngraph::element::f16;
        } else if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_INT32)) {
            elementType = ngraph::element::i32;
        } else if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
            auto convertInput =
                std::make_shared<ngraph::opset3::Convert>(input, ngraph::element::i32);
            input = std::make_shared<ngraph::opset3::Clamp>(convertInput, 0, 255);
            elementType = ngraph::element::u8;
        } else if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM_SIGNED) ||
                   checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_SYMM)) {
            auto convertInput =
                std::make_shared<ngraph::opset3::Convert>(input, ngraph::element::i32);
            input = std::make_shared<ngraph::opset3::Clamp>(convertInput, -128, 127);
            elementType = ngraph::element::i8;
        } else if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT16_ASYMM)) {
            auto convertInput =
                std::make_shared<ngraph::opset3::Convert>(input, ngraph::element::i32);
            input = std::make_shared<ngraph::opset3::Clamp>(convertInput, 0, 65535);
            elementType = ngraph::element::u16;
        } else if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT16_SYMM)) {
            auto convertInput =
                std::make_shared<ngraph::opset3::Convert>(input, ngraph::element::i32);
            input = std::make_shared<ngraph::opset3::Clamp>(convertInput, -32768, 32767);
            elementType = ngraph::element::i16;
        }
        outputNode = std::make_shared<ngraph::opset3::Convert>(input, elementType);
    }

    mNgraphNodes->setOutputAtOperandIndex(outputIndex, outputNode);
    const auto op = sModelInfo->getOperand(outputIndex);
    if (op.lifetime == OperandLifeTime::SUBGRAPH_OUTPUT) {
        addResultNode(mDefaultOutputIndex, outputNode);
    }

    return nullptr;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
