#include <ReduceSum.hpp>
#undef LOG_TAG
#define LOG_TAG "ReduceSum"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

ReduceSum::ReduceSum(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ngraph::Node> ReduceSum::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);
    auto reduction_axes = getInputNode(1);
    auto keep_dims = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 2);

    auto outputNode = std::make_shared<ngraph::opset3::ReduceSum>(input, reduction_axes, keep_dims);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
