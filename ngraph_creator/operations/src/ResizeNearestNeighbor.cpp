#include <ResizeNearestNeighbor.hpp>
#undef LOG_TAG
#define LOG_TAG "ResizeNearestNeighbor"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

ResizeNearestNeighbor::ResizeNearestNeighbor(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool ResizeNearestNeighbor::validate() {
    // TODO Add FLOAT16 check when VPUX plugin is supported
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        ALOGE("%s check for output types failed", __func__);
        return false;
    }

    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    const auto& inputDimensionsSize = getInputOperandDimensions(0).size();
    if (inputDimensionsSize != 4) {
        ALOGE("%s Invalid dimensions size for input(%lu)", __func__, inputDimensionsSize);
        return false;
    }
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> ResizeNearestNeighbor::createNode() {
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);

    std::shared_ptr<ngraph::Node> outputNode;
    int32_t input_width = 0, input_height = 0;
    bool useNchw = false;
    int32_t layout = 0;
    float width_scale = 0.0f, height_scale = 0.0f;
    bool align_corners = false, half_pixel = false;
    const auto& inputDimensions = getInputOperandDimensions(0);
    int32_t out_width = 0, out_height = 0;

    std::shared_ptr<ngraph::Node> inputNode;
    struct ngraph::op::v4::Interpolate::InterpolateAttrs attrs;

    inputNode = getInputNode(0);
    switch (inputsSize) {
        case 6:
            half_pixel = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 5);
            __attribute__((fallthrough));
        case 5:
            align_corners = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 4);
            __attribute__((fallthrough));
        case 4:
            layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 3);
            __attribute__((fallthrough));
        default:
            break;
    }

    if (layout) useNchw = true;
    if (useNchw) {
        input_width = inputDimensions[3];
        input_height = inputDimensions[2];
    } else {
        input_width = inputDimensions[2];
        input_height = inputDimensions[1];
    }

    if (!useNchw) inputNode = transpose(NHWC_NCHW, inputNode);
    // FLOAT16 type check added for future when VPUX plugin support is added
    if (checkInputOperandType(1, (int32_t)OperandType::FLOAT32) ||
        checkInputOperandType(1, (int32_t)OperandType::FLOAT16)) {
        // In tensorflow lite, resizing by size is supported. Scaling factors are
        // calculated based on output shape.
        attrs.shape_calculation_mode = ngraph::op::v4::Interpolate::ShapeCalcMode::sizes;
        width_scale = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 1);
        height_scale = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 2);
        out_width = (int)(input_width * width_scale);
        out_height = (int)(input_height * height_scale);
        // Recalculating scaling factors here because of typecasting output shape to
        // integer
        width_scale = (float)out_width / (float)input_width;
        height_scale = (float)out_height / (float)input_height;
    } else if (checkInputOperandType(1, (int32_t)OperandType::INT32)) {
        attrs.shape_calculation_mode = ngraph::op::v4::Interpolate::ShapeCalcMode::sizes;
        out_width = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);
        out_height = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 2);
        width_scale = (float)out_width / (float)input_width;
        height_scale = (float)out_height / (float)input_height;
    }

    if (align_corners == true) {
        attrs.coordinate_transformation_mode =
            ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners;
    } else if (half_pixel == true) {
        attrs.coordinate_transformation_mode =
            ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel;
    } else {
        // If none of the align_corners and half_pixel are true, transformation
        // mode is set to asymmetric
        attrs.coordinate_transformation_mode =
            ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric;
    }

    // mode is passed as "nearest" for Nearest Neighbor interpolation
    attrs.mode = ngraph::op::v4::Interpolate::InterpolateMode::nearest;
    attrs.nearest_mode = ngraph::op::v4::Interpolate::NearestMode::floor;

    std::vector<int32_t> output_shape = {out_height, out_width};
    auto outputShapeNode = createConstNode(ngraph::element::i32, {2}, output_shape);

    std::vector<float> scale_vec = {height_scale, width_scale};
    std::shared_ptr<ngraph::Node> scaleNode;
    if (checkInputOperandType(1, (int32_t)OperandType::FLOAT16))
        scaleNode = createConstNode(ngraph::element::f16, {2}, scale_vec);
    else if (checkInputOperandType(1, (int32_t)OperandType::FLOAT32) ||
             checkInputOperandType(1, (int32_t)OperandType::INT32)) {
        scaleNode = createConstNode(ngraph::element::f32, {2}, scale_vec);
    }

    std::vector<int32_t> axes_vec = {2, 3};
    auto axesNode = createConstNode(ngraph::element::i32, {2}, axes_vec);

    outputNode = std::make_shared<ngraph::op::v4::Interpolate>(inputNode, outputShapeNode,
                                                               scaleNode, axesNode, attrs);
    if (!useNchw) {
        outputNode = transpose(NCHW_NHWC, outputNode);
    }
    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
