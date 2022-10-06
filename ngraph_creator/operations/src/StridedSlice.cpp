#include <StridedSlice.hpp>
#undef LOG_TAG
#define LOG_TAG "StridedSlice"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

StridedSlice::StridedSlice(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool StridedSlice::validate() {
    // Check input rank
    const int64_t inputRank = getInputOperandDimensions(0).size();
    if (inputRank > 4) {
        ALOGE("%s Invalid input dimensions size!", __func__);
        return false;
    }

    // TODO: Add Support for all_tensors_as_inputs
    auto& begins_OperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
    if (!sModelInfo->isOperandLifeTimeConst(begins_OperandIndex)) {
        ALOGE("%s Only Constant dimensions supported now", __func__);
        return false;
    }

    auto ends_OperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 2);
    if (!sModelInfo->isOperandLifeTimeConst(ends_OperandIndex)) {
        ALOGE("%s Only Constant dimensions supported now", __func__);
        return false;
    }

    auto& strides_OperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 3);
    if (!sModelInfo->isOperandLifeTimeConst(strides_OperandIndex)) {
        ALOGE("%s Only Constant dimensions supported now", __func__);
        return false;
    }

    auto shrink_axis_mask = sModelInfo->ParseOperationInput<int32_t>(mNnapiOperationIndex, 6);
    std::vector<int64_t> shrink_axis_mask_bits = getMaskBits(shrink_axis_mask, inputRank);

    for (int i = 0; i < inputRank; i++) {
        if (shrink_axis_mask_bits[i]) {
            // Check for negative stride when shrink axis bit is set
            auto stridesVector = sModelInfo->GetConstVecOperand<int32_t>(strides_OperandIndex);
            if (stridesVector[i] < 0) {
                ALOGE("%s Negative stride value when shrink axis bit set is not supported",
                      __func__);
                return false;
            }

            // check for slice size larger than expected output
            auto beginVector = sModelInfo->GetConstVecOperand<int32_t>(begins_OperandIndex);
            auto endVector = sModelInfo->GetConstVecOperand<int32_t>(ends_OperandIndex);
            if (((beginVector[i] - endVector[i]) > 1) || ((beginVector[i] - endVector[i]) < -1)) {
                ALOGE("%s Trying to access invalid slice size when shrink axis bit is set",
                      __func__);
                return false;
            }
        }
    }

    return true;
}

std::shared_ptr<ov::Node> StridedSlice::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> data = getInputNode(0);
    std::shared_ptr<ov::Node> begin = getInputNode(1);
    std::shared_ptr<ov::Node> end = getInputNode(2);
    std::shared_ptr<ov::Node> strides = getInputNode(3);

    auto begin_mask = sModelInfo->ParseOperationInput<int32_t>(mNnapiOperationIndex, 4);
    auto end_mask = sModelInfo->ParseOperationInput<int32_t>(mNnapiOperationIndex, 5);
    auto shrink_axis_mask = sModelInfo->ParseOperationInput<int32_t>(mNnapiOperationIndex, 6);

    const auto data_dim_size = getInputOperandDimensions(0).size();
    std::vector<int64_t> begin_mask_bits, end_mask_bits, shrink_axis_mask_bits;

    begin_mask_bits = getMaskBits(begin_mask, data_dim_size);
    end_mask_bits = getMaskBits(end_mask, data_dim_size);
    shrink_axis_mask_bits = getMaskBits(shrink_axis_mask, data_dim_size);
    const std::vector<int64_t> new_axis_mask = std::vector<int64_t>{};
    const std::vector<int64_t> ellipsis_mask = std::vector<int64_t>{};

    std::shared_ptr<ov::Node> outputNode = std::make_shared<ov::opset3::StridedSlice>(
        data, begin, end, strides, begin_mask_bits, end_mask_bits, new_axis_mask,
        shrink_axis_mask_bits, ellipsis_mask);

    return outputNode;
}

std::vector<int64_t> StridedSlice::getMaskBits(int32_t maskValue, size_t vec_size) {
    std::vector<int64_t> mask_bits(vec_size);
    int i = 0;
    while (maskValue != 0) {
        mask_bits[i] = (maskValue % 2) == 0 ? 0 : 1;
        maskValue /= 2;
        i++;
    }
    return mask_bits;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
