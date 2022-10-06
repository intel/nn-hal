#include <HardSwish.hpp>
#define LOG_TAG "HardSwish"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

HardSwish::HardSwish(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool HardSwish::validate() {
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ov::Node> HardSwish::createNode() {
    std::shared_ptr<ov::Node> outputNode, inputNode;
    inputNode = getInputNode(0);

    outputNode = std::make_shared<ov::op::v4::HSwish>(inputNode);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
