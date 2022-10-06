#include <Pow.hpp>
#undef LOG_TAG
#define LOG_TAG "Pow"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Pow::Pow(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> Pow::createNode() {
    // Creating input nodes
    auto base = getInputNode(0);
    auto exponent = getInputNode(1);

    auto outputNode =
        std::make_shared<ov::opset3::Power>(base, exponent, ov::op::AutoBroadcastType::NUMPY);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
