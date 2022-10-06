#include <ReduceMax.hpp>
#undef LOG_TAG
#define LOG_TAG "ReduceMax"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

ReduceMax::ReduceMax(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> ReduceMax::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);
    auto reduction_axes = getInputNode(1);
    auto keep_dims = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 2);

    std::shared_ptr<ov::Node> outputNode;
    outputNode = std::make_shared<ov::opset3::ReduceMax>(input, reduction_axes, keep_dims);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
