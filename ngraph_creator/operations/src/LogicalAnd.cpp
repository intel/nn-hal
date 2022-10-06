#include <LogicalAnd.hpp>
#undef LOG_TAG
#define LOG_TAG "LogicalAnd"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

LogicalAnd::LogicalAnd(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> LogicalAnd::createNode() {
    // Creating input nodes
    auto input1 = getInputNode(0);
    auto input2 = getInputNode(1);

    auto outputNode =
        std::make_shared<ov::opset3::LogicalAnd>(input1, input2, ov::op::AutoBroadcastType::NUMPY);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
