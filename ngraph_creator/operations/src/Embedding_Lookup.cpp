#include <Embedding_Lookup.hpp>
#undef LOG_TAG
#define LOG_TAG "Embedding_Lookup"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Embedding_Lookup::Embedding_Lookup(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Embedding_Lookup::validate() {
    const auto inputRank = getInputOperandDimensions(1).size();
    if (inputRank < 2) return false;

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Embedding_Lookup::createNode() {
    // Creating input nodes
    auto indices = getInputNode(0);
    auto input = getInputNode(1);

    auto axis = createConstNode(ngraph::element::i32, {}, std::vector<int64_t>{0});
    auto outputNode = std::make_shared<ngraph::opset3::Gather>(input, indices, axis);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
