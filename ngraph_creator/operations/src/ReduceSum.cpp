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

bool ReduceSum::validate() {
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

    // TODO: Add Support for all_tensors_as_inputs
    if (!sModelInfo->isOperandLifeTimeConst(input_OperandIndex) ||
        !sModelInfo->isOperandLifeTimeConst(dim_reduce_OperandIndex)) {
        ALOGE("%s Only Constant dimensions supported now", __func__);
        return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
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
