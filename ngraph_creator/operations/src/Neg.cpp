#include <Neg.hpp>
#undef LOG_TAG
#define LOG_TAG "Neg"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Neg::Neg(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ngraph::Node> Neg::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;

    input = getInputNode(0);

    auto outputNode = std::make_shared<ngraph::opset3::Negative>(input);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
