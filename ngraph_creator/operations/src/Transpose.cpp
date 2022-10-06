#include <Transpose.hpp>
#undef LOG_TAG
#define LOG_TAG "Transpose"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Transpose::Transpose(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Transpose::validate() {
    // TODO: Add Support for all_tensors_as_inputs
    const auto& dimsOperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
    const auto& dims = getInputOperandDimensions(1);
    if (!dims.empty() && dims[0] != 0 && !sModelInfo->isOperandLifeTimeConst(dimsOperandIndex)) {
        ALOGE("%s Only Constant dimensions supported now", __func__);
        return false;
    }

    return true;
}

std::shared_ptr<ov::Node> Transpose::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input;

    input = getInputNode(0);

    std::shared_ptr<ov::Node> order;

    const auto& dims = getInputOperandDimensions(1);
    if (!dims.empty() && dims[0] != 0) {
        order = getInputNode(1);
    } else {
        order = createConstNode(ov::element::i32, {0}, convertToVector(0));
    }

    std::shared_ptr<ov::Node> outputNode;

    outputNode = std::make_shared<ov::opset3::Transpose>(input, order);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
