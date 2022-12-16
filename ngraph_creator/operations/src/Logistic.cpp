#include <Logistic.hpp>
#undef LOG_TAG
#define LOG_TAG "Logistic"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Logistic::Logistic(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> Logistic::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input;

    input = getInputNode(0);

    std::shared_ptr<ov::Node> outputNode;
    outputNode = std::make_shared<ov::opset3::Sigmoid>(input);

    // TODO : This is a WA for selfie_segmentation model
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
