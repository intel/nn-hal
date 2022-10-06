#include <Dequantize.hpp>
#undef LOG_TAG
#define LOG_TAG "Dequantize"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Dequantize::Dequantize(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> Dequantize::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input, outputNode;
    input = getInputNode(0, false);
    const auto& inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);

    if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT16))
        outputNode = DequantizeNode(input, inputIndex, ov::element::f16);
    else
        outputNode = DequantizeNode(input, inputIndex, ov::element::f32);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
