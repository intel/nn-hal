#include <Relu6.hpp>
#undef LOG_TAG
#define LOG_TAG "Relu6"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Relu6::Relu6(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> Relu6::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input;

    input = getInputNode(0);

    std::shared_ptr<ov::Node> outputNode;

    outputNode = std::make_shared<ov::opset3::Clamp>(input, 0, 6);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
