#include <Dequantize.hpp>
#define LOG_TAG "Dequantize"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Dequantize::Dequantize(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Dequantize::validate() {
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Dequantize::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input, outputNode;
    input = getInputNode(0, false);
    const auto& inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);

    outputNode = DequantizeNode(input, inputIndex, ngraph::element::f32);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
