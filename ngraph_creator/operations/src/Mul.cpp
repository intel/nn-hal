#include <Mul.hpp>
#undef LOG_TAG
#define LOG_TAG "Mul"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Mul::Mul(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ngraph::Node> Mul::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input1, input2;

    input1 = getInputNode(0);
    input2 = getInputNode(1);

    auto activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);

    auto mulNode = std::make_shared<ngraph::opset3::Multiply>(input1, input2,
                                                              ngraph::op::AutoBroadcastType::NUMPY);

    auto outputNode = applyActivation(mulNode, activationFn);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
