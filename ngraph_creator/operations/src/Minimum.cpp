#include <Minimum.hpp>
#undef LOG_TAG
#define LOG_TAG "Minimum"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Minimum::Minimum(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> Minimum::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input1, input2;

    input1 = getInputNode(0);
    input2 = getInputNode(1);

    std::shared_ptr<ov::Node> outputNode;

    outputNode =
        std::make_shared<ov::opset3::Minimum>(input1, input2, ov::op::AutoBroadcastType::NUMPY);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
