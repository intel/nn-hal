#include <Tanh.hpp>
#undef LOG_TAG
#define LOG_TAG "Tanh"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Tanh::Tanh(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> Tanh::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input;

    input = getInputNode(0);

    std::shared_ptr<ov::Node> outputNode;

    outputNode = std::make_shared<ov::opset3::Tanh>(input);
    //This is required where Tanh is the final node in the graph to convert back to
    //NHWC format
    auto outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
    const auto op = sModelInfo->getOperand(outputIndex);
    if (op.lifetime == OperandLifeTime::SUBGRAPH_OUTPUT) {
        outputNode = transpose(NCHW_NHWC, outputNode);
    }

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
