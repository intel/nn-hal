#include <Abs.hpp>
#undef LOG_TAG
#define LOG_TAG "Abs"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Abs::Abs(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> Abs::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);

    auto outputNode = std::make_shared<ov::opset3::Abs>(input);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
