#include <Select.hpp>
#define LOG_TAG "Select"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Select::Select(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Select::validate() {
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Select::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input1, input2, input3;

    input1 = getInputNode(0);
    input2 = getInputNode(1);
    input3 = getInputNode(2);

    std::shared_ptr<ngraph::Node> outputNode;

    outputNode = std::make_shared<ngraph::opset3::Select>(input1, input2, input3);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
