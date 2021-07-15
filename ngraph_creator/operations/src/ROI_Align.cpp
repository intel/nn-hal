#include <ROI_Align.hpp>
#define LOG_TAG "ROI_Align"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

ROI_Align::ROI_Align(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool ROI_Align::validate() {
    ALOGV("%s Entering", __func__);

    // Check Output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        ALOGE("%s Output operand 0 is not of type FP32. Unsupported operation", __func__);
        return false;
    }

    // Check Input Type
#if __ANDROID__
    // TODO: Issue reported in Android with OV_2021.2 version. Remove this when OV is upgraded
    const auto& inputDimensions = getInputOperandDimensions(0);
    auto layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 9);
    size_t height, width;
    if (layout) {
        height = inputDimensions[2];
        width = inputDimensions[3];
    } else {
        height = inputDimensions[1];
        width = inputDimensions[2];
    }
    if (height != width) {
        ALOGE("%s Not handled with different height and width", __func__);
        return false;
    }
#endif

    if (isZeroSizedInput(0) || isZeroSizedInput(1) || isZeroSizedInput(2)) {
        ALOGE("%s Not handling zero sized input for dimension 0", __func__);
        return false;
    }

    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        ALOGE("%s Input operand 0 is not of type FP32. Unsupported operation", __func__);
        return false;
    }
    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_FLOAT32)) {
        ALOGE("%s Input operand 1 is not of type FP32. Unsupported operation", __func__);
        return false;
    }

    // TODO: support for different height_ratio and width_ratio
    // values
    auto height_ratio = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 5);
    auto width_ratio = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 6);
    if (height_ratio != width_ratio) {
        ALOGE(
            "%s: Ratio of Height and Ratio of Width from orginal image to feature map must be same "
            "for ROI Align. Got %f and %f",
            __func__, height_ratio, width_ratio);
        return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> ROI_Align::createNode() {
    ALOGV("%s Entering", __func__);

    bool useNchw = false;

    // Read inputs
    auto feat_maps = getInputNode(0);      // 4D tensor
    auto rois = getInputNode(1);           // 2D tensor
    auto batch_indices = getInputNode(2);  // 1D tensor
    auto output_height = sModelInfo->ParseOperationInput<int32_t>(
        mNnapiOperationIndex, 3);  // height of the output tensor
    auto output_width = sModelInfo->ParseOperationInput<int32_t>(mNnapiOperationIndex,
                                                                 4);  // width of the output tensor
    auto height_ratio = sModelInfo->ParseOperationInput<float>(
        mNnapiOperationIndex,
        5);  // ratio from the height of original image to the height of feature map.
    //TODO: Since same ratio is applied on height and width, commenting
    //the width_ratio and sampling_pts_w currently
    //auto width_ratio = sModelInfo->ParseOperationInput<float>(
    //    mNnapiOperationIndex,
    //    6);  // ratio from the width of original image to the height of feature map.
    auto sampling_pts_h = sModelInfo->ParseOperationInput<int32_t>(mNnapiOperationIndex, 7);
    //auto sampling_pts_w = sModelInfo->ParseOperationInput<int32_t>(mNnapiOperationIndex, 8);
    auto layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 9);

    if (layout) useNchw = true;

    if (!useNchw)  // No conversion needed if useNchw set
        feat_maps = transpose(NHWC_NCHW, feat_maps);

    float spatial_scale = 1.0 / (height_ratio);
    int sampling_ratio = sampling_pts_h;

    std::shared_ptr<ngraph::Node> outputNode = std::make_shared<ngraph::opset3::ROIAlign>(
        feat_maps, rois, batch_indices, output_height, output_width, sampling_ratio, spatial_scale,
        "avg");

    if (!useNchw) outputNode = transpose(NCHW_NHWC, outputNode);

    ALOGV("%s PASSED", __func__);

    return outputNode;
}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
