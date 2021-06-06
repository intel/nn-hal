#include <Split.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Split::Split(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Split::validate() {
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

    if (!checkInputOperandType(2, (int32_t)OperandType::INT32)) {
        return false;
    }

    return true;
}

void Split::connectOperationToGraph() { createNode(); }

std::shared_ptr<ngraph::Node> Split::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> splitNode;

    splitNode = getInputNode(0, false);

    auto axis = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);
    auto axisNode = createConstNode(ngraph::element::i32, {}, convertToVector(axis));
    auto numSplits = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);

    auto outputNode =
        std::make_shared<ngraph::opset3::Split>(splitNode, axisNode, numSplits)->outputs();

    for (int i = 0; i < numSplits; i++) {
        auto outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, i);
        // TODO: remove this dummy convert
        std::shared_ptr<ngraph::Node> outNode;
        if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
            outNode =
                std::make_shared<ngraph::opset3::Convert>(outputNode[i], ngraph::element::f32);
        } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_INT32)) {
            outNode =
                std::make_shared<ngraph::opset3::Convert>(outputNode[i], ngraph::element::i32);
        } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
            outNode = std::make_shared<ngraph::opset3::Convert>(outputNode[i], ngraph::element::u8);
        }

        mNgraphNodes->setOutputAtOperandIndex(outputIndex, outNode);
        const auto op = sModelInfo->getOperand(outputIndex);
        if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
            addResultNode(outputIndex, outNode);
        }
    }

    return nullptr;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
