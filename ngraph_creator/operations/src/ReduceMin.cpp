#include <ReduceMin.hpp>
#undef LOG_TAG
#define LOG_TAG "ReduceMin"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

ReduceMin::ReduceMin(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ngraph::Node> ReduceMin::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;

    input = getInputNode(0);

    auto reduction_axes = getInputNode(1);
    auto keep_dims = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 2);

    std::shared_ptr<ngraph::Node> outputNode;
    outputNode = std::make_shared<ngraph::opset3::ReduceMin>(input, reduction_axes, keep_dims);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
