#include <ROIPooling.hpp>
#undef LOG_TAG
#define LOG_TAG "ROIPooling"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

ROIPooling::ROIPooling(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool ROIPooling::validate() {
    ALOGV("%s Entering", __func__);

    // Check Output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        ALOGE("%s Output operand 0 is not of type FP32. Unsupported operation", __func__);
        return false;
    }

    // Check Input Type
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
            "for ROI Pooling. Got %f and %f",
            __func__, height_ratio, width_ratio);
        return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> ROIPooling::createNode() {
    ALOGV("%s Entering", __func__);

    bool useNchw = false;

    // Read inputs
    auto feat_maps = getInputNode(0);  // 4D tensor
    auto output_height = sModelInfo->ParseOperationInput<int32_t>(
        mNnapiOperationIndex, 3);  // height of the output tensor
    auto output_width = sModelInfo->ParseOperationInput<int32_t>(mNnapiOperationIndex,
                                                                 4);  // width of the output tensor
    auto height_ratio = sModelInfo->ParseOperationInput<float>(
        mNnapiOperationIndex,
        5);  // ratio from the height of original image to the height of feature map.
    auto layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 7);

    if (layout) useNchw = true;

    if (!useNchw)  // No conversion needed if useNchw set
        feat_maps = transpose(NHWC_NCHW, feat_maps);

    auto output_size = ngraph::Shape{(size_t)output_height, (size_t)output_width};
    float spatial_scale = 1.0 / (height_ratio);

    // Concat batch index of shape[num_rois] and rois shape[num_rois, 4]
    // to create 2-D Tensor of shape[num_rois, 5] => bi,x1,y1,x2,y2
    std::vector<ngraph::Output<ngraph::Node>> inputs;
    auto axis = 1;
    // add bi node to inputs for concat
    const auto& biOperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 2);
    auto bi_vec = sModelInfo->GetConstVecOperand<int32_t>(biOperandIndex);
    const auto bi_node =
        createConstNode(ngraph::element::f32, ngraph::Shape{bi_vec.size(), 1}, bi_vec);
    inputs.push_back(bi_node);
    // add rois node to inputs for concat
    auto inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
    auto inputOp = mNgraphNodes->getOperationOutput(inputIndex);
    inputs.push_back(inputOp);

    std::shared_ptr<ngraph::Node> roiNode = std::make_shared<ngraph::opset3::Concat>(inputs, axis);
    ALOGI("%s Concatinated roi_node created", __func__);

    std::shared_ptr<ngraph::Node> outputNode = std::make_shared<ngraph::opset3::ROIPooling>(
        feat_maps, roiNode, output_size, spatial_scale);

    if (!useNchw) outputNode = transpose(NCHW_NHWC, outputNode);

    ALOGV("%s PASSED", __func__);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
