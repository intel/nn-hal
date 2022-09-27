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

bool ReduceMin::validate() {
    // Check input rank
    const auto inputRank = getInputOperandDimensions(0).size();

    if (inputRank > 4) 
        return false;

    if ( !isValidInputTensor(0) || !isValidInputTensor(1)) {
         ALOGE("%s Empty  or Invalid dimensions size for input", __func__);
         return false;
    }

    auto& input_OperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
    auto& dim_reduce_OperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);

    ALOGV("%s PASSED", __func__);
    return true;
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
