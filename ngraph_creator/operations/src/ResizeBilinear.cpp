#include <ResizeBilinear.hpp>
#define LOG_TAG "ResizeBilinear"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

ResizeBilinear::ResizeBilinear(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool ResizeBilinear::validate() {
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
    for (int i = 1; i < 3; i++) {
        if (!checkInputOperandType(i, (int32_t)OperandType::FLOAT32) &&
            !checkInputOperandType(i, (int32_t)OperandType::INT32)) {
            const auto& operandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, i);
            const auto operandType = (int32_t)sModelInfo->getOperandType(operandIndex);
            ALOGE("%d =====> %s input index %d is of type %d", i, __func__, operandIndex,
                  operandType);
            ALOGE("%s check for input float types failed", __func__);
            return false;
        }
    }

    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);

    if (inputsSize >= 4) {
        if (!checkInputOperandType(3, (int32_t)OperandType::BOOL)) {
            return false;
        }
    }

    if (inputsSize >= 5) {
        if (!checkInputOperandType(4, (int32_t)OperandType::BOOL)) {
            return false;
        }
        if (inputsSize == 6) {
            if (!checkInputOperandType(5, (int32_t)OperandType::BOOL)) {
                return false;
            }
        }
    }
    const auto& inputDimensionsSize = getInputOperandDimensions(0).size();
    if (inputDimensionsSize != 4) {
        ALOGE("%s Invalid dimensions size for input(%d)", __func__, inputDimensionsSize);
        return false;
    }
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> ResizeBilinear::createNode() {
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);

    std::shared_ptr<ngraph::Node> outputNode;
    int32_t input_width, input_height;
    bool useNchw = false;
    int32_t layout = 0;
    float width_scale, height_scale;
    bool align_corners = false, half_pixel = false;
    const auto& inputDimensions = getInputOperandDimensions(0);

    std::shared_ptr<ngraph::Node> inputNode;
    struct ngraph::op::v4::Interpolate::InterpolateAttrs attrs;

    inputNode = getInputNode(0);
    if (inputsSize >= 4) {
        layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 3);
    }
    if (layout) useNchw = true;
    if (useNchw) {
        input_width = inputDimensions[3];
        input_height = inputDimensions[2];
    } else {
        input_width = inputDimensions[2];
        input_height = inputDimensions[1];
    }

    // FLOAT16 type check added for future when VPUX plugin support is added
    if (checkInputOperandType(1, (int32_t)OperandType::FLOAT32) ||
        checkInputOperandType(1, (int32_t)OperandType::FLOAT16)) {
        attrs.shape_calculation_mode = ngraph::op::v4::Interpolate::ShapeCalcMode::scales;
        width_scale = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 1);
        height_scale = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 2);
    } else if (checkInputOperandType(1, (int32_t)OperandType::INT32)) {
        attrs.shape_calculation_mode = ngraph::op::v4::Interpolate::ShapeCalcMode::scales;
        int32_t out_width = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);
        int32_t out_height = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 2);
        width_scale = (float)out_width / (float)input_width;
        height_scale = (float)out_height / (float)input_height;
    }

    if (inputsSize >= 5) {
        align_corners = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 4);
        if (inputsSize == 6)
            half_pixel = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 5);
    }

    if (align_corners == true) {
        attrs.coordinate_transformation_mode =
            ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners;
    } else if (half_pixel == true) {
        attrs.coordinate_transformation_mode =
            ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel;
    } else {
        attrs.coordinate_transformation_mode =
            ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners;
    }

    // mode is passed as "linear" for bilinear interpolation
    attrs.mode = ngraph::op::v4::Interpolate::InterpolateMode::linear;

    const auto& inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
    if (mNgraphNodes->isForcedNchw(inputIndex)) {
        if (useNchw) {
            ALOGI("%s Forced NCHW done already but NCHW flag set at operationIndex %d", __func__,
                  mNnapiOperationIndex);
            inputNode = transpose(NCHW_NHWC, inputNode);
            mNgraphNodes->setForcedNchw(mDefaultOutputIndex, false);
        } else {
            // Already forced NCHW, propogate the flag
            mNgraphNodes->setForcedNchw(mDefaultOutputIndex, true);
        }
    } else if (!useNchw) {  // No conversion needed if useNchw set
        inputNode = transpose(NHWC_NCHW, inputNode);
        mNgraphNodes->setForcedNchw(mDefaultOutputIndex, true);
        ALOGD("%s Forced NCHW conversion at operationIndex %d", __func__, mNnapiOperationIndex);
    }

    std::vector<float> scaleFactors = {1.0f, 1.0f, width_scale, height_scale};
    std::shared_ptr<ngraph::Node> scaleConstNode, shape;
    // TODO: VPU Plugin may not support all type of conversions. Update this
    // section to support VPUX Plugin
    if (checkInputOperandType(1, (int32_t)OperandType::FLOAT16)) {
        shape = std::make_shared<ngraph::opset3::Convert>(
            std::make_shared<ngraph::opset3::ShapeOf>(inputNode), ngraph::element::f16);
        scaleConstNode = createConstNode(ngraph::element::f16, {4}, scaleFactors);
    } else if (checkInputOperandType(1, (int32_t)OperandType::FLOAT32) ||
               checkInputOperandType(1, (int32_t)OperandType::INT32)) {
        shape = std::make_shared<ngraph::opset3::Convert>(
            std::make_shared<ngraph::opset3::ShapeOf>(inputNode), ngraph::element::f32);
        scaleConstNode = createConstNode(ngraph::element::f32, {4}, scaleFactors);
    }
    // Scale input shape to calculate final output shape
    auto outputShape = std::make_shared<ngraph::op::v1::Multiply>(scaleConstNode, shape);

    // TODO: VPU Plugin may not support all type of conversions. Update this
    // section to support VPUX Plugin
    auto outputShape_int32 =
        std::make_shared<ngraph::opset3::Convert>(outputShape, ngraph::element::i32);

    auto begin_index = createConstNode(ngraph::element::i32, {1}, convertToVector(1));
    auto end_index = createConstNode(ngraph::element::i32, {1}, convertToVector(4));
    auto strides = createConstNode(ngraph::element::i32, {1}, convertToVector(2));

    int64_t begin_mask = 0;
    int64_t end_mask = 0;
    // Get output shape(width x height) from scaled Shape vector
    auto outputShapeNode = std::make_shared<ngraph::op::v1::StridedSlice>(
        outputShape_int32, begin_index, end_index, strides, convertToVector(begin_mask),
        convertToVector(end_mask));

    std::vector<float> scale_vec = {width_scale, height_scale};
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
        mNgraphNodes->setForcedNchw(mDefaultOutputIndex, false);
    }
    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
