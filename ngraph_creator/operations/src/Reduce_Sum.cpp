#include <Reduce_Sum.hpp>
#undef LOG_TAG
#define LOG_TAG "Reduce_Sum"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Reduce_Sum::Reduce_Sum(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Reduce_Sum::validate() {
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Reduce_Sum::createNode() {
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
