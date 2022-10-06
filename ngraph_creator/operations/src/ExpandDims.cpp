#include <ExpandDims.hpp>
#undef LOG_TAG
#define LOG_TAG "ExpandDims"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

ExpandDims::ExpandDims(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool ExpandDims::validate() {
    // Check input rank
    const auto inputRank = getInputOperandDimensions(0).size();
    if (inputRank < 1) return false;

    return true;
}

std::shared_ptr<ov::Node> ExpandDims::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);
    auto index = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);

    auto axes = createConstNode(ov::element::i32, {}, convertToVector(index));

    auto outputNode = std::make_shared<ov::opset3::Unsqueeze>(input, axes);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
