#include <Equal.hpp>
#undef LOG_TAG
#define LOG_TAG "Equal"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Equal::Equal(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Equal::validate() {
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Equal::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input1, input2;

    input1 = getInputNode(0);
    input2 = getInputNode(1);

    std::shared_ptr<ngraph::Node> outputNode;
    outputNode = std::make_shared<ngraph::opset3::Equal>(input1, input2,
                                                         ngraph::op::AutoBroadcastType::NUMPY);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
