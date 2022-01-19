#include <LogicalNot.hpp>
#undef LOG_TAG
#define LOG_TAG "LogicalNot"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

LogicalNot::LogicalNot(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ngraph::Node> LogicalNot::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);

    auto outputNode = std::make_shared<ngraph::opset3::LogicalNot>(input);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
