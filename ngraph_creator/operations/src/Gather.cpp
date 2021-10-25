#include <Gather.hpp>
#undef LOG_TAG
#define LOG_TAG "Gather"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Gather::Gather(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Gather::validate() {
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Gather::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> gatherVals;

    gatherVals = getInputNode(0);

    // axis range [-n, n]
    auto axis = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);
    auto axisNode = createConstNode(ngraph::element::i32, {}, convertToVector(axis));

    auto indices = getInputNode(2);

    std::shared_ptr<ngraph::Node> outputNode;
    outputNode = std::make_shared<ngraph::opset3::Gather>(gatherVals, indices, axisNode);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
