#include <Select.hpp>
#undef LOG_TAG
#define LOG_TAG "Select"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Select::Select(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> Select::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input1, input2, input3;

    input1 = getInputNode(0);
    input2 = getInputNode(1);
    input3 = getInputNode(2);

    std::shared_ptr<ov::Node> outputNode;

    outputNode = std::make_shared<ov::opset3::Select>(input1, input2, input3);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
