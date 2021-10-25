#include <Logical_Not.hpp>
#undef LOG_TAG
#define LOG_TAG "Logical_Not"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Logical_Not::Logical_Not(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Logical_Not::validate() {
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Logical_Not::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);

    auto outputNode = std::make_shared<ngraph::opset3::LogicalNot>(input);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
