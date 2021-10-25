#include <Less_Equal.hpp>
#undef LOG_TAG
#define LOG_TAG "Less_Equal"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Less_Equal::Less_Equal(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Less_Equal::validate() {
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Less_Equal::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input1, input2;

    input1 = getInputNode(0);
    input2 = getInputNode(1);

    std::shared_ptr<ngraph::Node> outputNode;

    outputNode = std::make_shared<ngraph::opset3::LessEqual>(input1, input2,
                                                             ngraph::op::AutoBroadcastType::NUMPY);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
