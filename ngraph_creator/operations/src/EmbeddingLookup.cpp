#include <EmbeddingLookup.hpp>
#undef LOG_TAG
#define LOG_TAG "EmbeddingLookup"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

EmbeddingLookup::EmbeddingLookup(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool EmbeddingLookup::validate() {
    const auto inputRank = getInputOperandDimensions(1).size();
    if (inputRank < 2) return false;

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ov::Node> EmbeddingLookup::createNode() {
    // Creating input nodes
    auto indices = getInputNode(0);
    auto input = getInputNode(1);

    auto axis = createConstNode(ov::element::i32, {}, std::vector<int64_t>{0});
    auto outputNode = std::make_shared<ov::opset3::Gather>(input, indices, axis);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
