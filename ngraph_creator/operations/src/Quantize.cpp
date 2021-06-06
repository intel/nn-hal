#include <Quantize.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Quantize::Quantize(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Quantize::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        return false;
    }

    return true;
}

void Quantize::connectOperationToGraph() { createNode(); }

std::shared_ptr<ngraph::Node> Quantize::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);

    auto inputElementType = input->get_element_type();

    const auto& outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);

    auto outputNode = QuantizeNode(input, outputIndex, ngraph::element::u8);

    mNgraphNodes->setOutputAtOperandIndex(outputIndex, outputNode);
    const auto op = sModelInfo->getOperand(outputIndex);
    if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
        addResultNode(mDefaultOutputIndex, outputNode);
    }

    return nullptr;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
