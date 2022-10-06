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

std::shared_ptr<ov::Node> Gather::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> gatherVals;

    gatherVals = getInputNode(0);

    // axis range [-n, n]
    auto axis = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);
    auto axisNode = createConstNode(ov::element::i32, {}, convertToVector(axis));

    auto indices = getInputNode(2);

    std::shared_ptr<ov::Node> outputNode;
    outputNode = std::make_shared<ov::opset3::Gather>(gatherVals, indices, axisNode);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
