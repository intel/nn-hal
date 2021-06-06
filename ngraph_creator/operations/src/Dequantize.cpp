#include <Dequantize.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Dequantize::Dequantize(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Dequantize::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_SYMM)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Dequantize::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;

    input = getInputNode(0, false);

    const auto& inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);

    auto outputNode = DequantizeNode(input, inputIndex, ngraph::element::f32);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
