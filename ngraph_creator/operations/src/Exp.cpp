#include <Exp.hpp>
#undef LOG_TAG
#define LOG_TAG "Exp"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Exp::Exp(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> Exp::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);

    auto outputNode = std::make_shared<ov::opset3::Exp>(input);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
