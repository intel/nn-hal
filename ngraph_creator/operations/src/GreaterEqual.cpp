#include <GreaterEqual.hpp>
#undef LOG_TAG
#define LOG_TAG "GreaterEqual"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

GreaterEqual::GreaterEqual(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> GreaterEqual::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input1, input2;

    input1 = getInputNode(0);
    input2 = getInputNode(1);

    std::shared_ptr<ov::Node> outputNode;
    outputNode = std::make_shared<ov::opset3::GreaterEqual>(input1, input2,
                                                            ov::op::AutoBroadcastType::NUMPY);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
