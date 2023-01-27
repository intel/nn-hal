#include <TransposeConv2D.hpp>
// Helper function
#include <NgraphHelper.hpp>
#undef LOG_TAG
#define LOG_TAG "TransposeConv2D"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

TransposeConv2D::TransposeConv2D(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool TransposeConv2D::validate() {
    // Check Input, Filter Dimension size
    const auto& inputDimensionsSize = getInputOperandDimensions(0).size();
    const auto& filterDimensionsSize = getInputOperandDimensions(1).size();
    if (inputDimensionsSize != 4 || filterDimensionsSize != 4) {
        ALOGE("%s Invalid dimensions size for input(%lu) or filter(%lu)", __func__,
              inputDimensionsSize, filterDimensionsSize);
        return false;
    }
    if (!isValidInputTensor(0) || !isValidInputTensor(1)) {
        ALOGE("%s Invalid dimensions for input or filter", __func__);
        return false;
    }

    if (checkInputOperandType(1, (int32_t)OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL)) {
        const auto& operandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
        const auto& operand = sModelInfo->getOperand(operandIndex);
        if (operand.extraParams.channelQuant().channelDim != 0) {
            return false;
        }
    }

    // TODO: Issue from OV 2021.4, remove this check once CVS-61723 is resolved
    // Workaround to ignore VTS large input error test cases
    const auto& inputDimensions = getInputOperandDimensions(0);

    if (inputDimensions[1] == 1 && inputDimensions[2] == 1 && inputDimensions[3] == 1) return false;

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> TransposeConv2D::createNode() {
    std::shared_ptr<ngraph::Node> inputNode;
    inputNode = getInputNode(0);
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    ALOGD("%s inputsSize %lu", __func__, inputsSize);

    bool isImplicit = false, isExplicit = false;

    if (inputsSize == 11) {
        isExplicit = true;
    } else if (inputsSize == 9) {
        isImplicit = true;
    } else {
        ALOGE("%s inputsSize %lu NOT SUPPORTED", __func__, inputsSize);
        return inputNode;
    }

    int32_t padding_left, padding_right;
    int32_t padding_top, padding_bottom;
    int32_t stride_width, stride_height;
    int32_t dilation_width_factor = 1, dilation_height_factor = 1;
    int32_t activationFn;
    int32_t layout = 0;
    int32_t padding_scheme;
    int32_t input_width, input_height;
    int32_t filter_width, filter_height;
    bool useNchw = false;
    std::vector<size_t> strides;
    std::vector<std::ptrdiff_t> pads_begin;
    std::vector<std::ptrdiff_t> pads_end;
    std::vector<size_t> dilations;
    ngraph::op::PadType auto_pad;

    std::shared_ptr<ngraph::Node> outputShapeNode = nullptr;

    const auto& inputDimensions = getInputOperandDimensions(0);

    {
        const auto& filterDimensions = getInputOperandDimensions(1);
        filter_width = filterDimensions[2];
        filter_height = filterDimensions[1];
    }

    if (isExplicit) {
        padding_left = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 3);
        padding_right = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 4);
        padding_top = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 5);
        padding_bottom = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 6);

        stride_width = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 7);
        stride_height = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 8);

        activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 9);
        layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 10);

        if (layout) useNchw = true;

        auto_pad = ngraph::op::PadType::EXPLICIT;
        {
            if (useNchw) {
                input_width = inputDimensions[3];
                input_height = inputDimensions[2];
            } else {
                input_width = inputDimensions[2];
                input_height = inputDimensions[1];
            }
        }
    }

    if (isImplicit) {
        const auto& outputShapeOperandIndex =
            sModelInfo->getOperationInput(mNnapiOperationIndex, 3);

        auto outputShape = sModelInfo->GetConstVecOperand<int32_t>(outputShapeOperandIndex);
        size_t spatial_dimensions_size = 2;
        std::vector<int32_t> spatial_dimensions(spatial_dimensions_size);

        padding_scheme = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 4);

        stride_width = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 5);
        stride_height = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 6);

        activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 7);
        layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 8);

        if (layout) useNchw = true;

        {
            if (useNchw) {
                spatial_dimensions[0] = outputShape[2];
                spatial_dimensions[1] = outputShape[3];
            } else {
                spatial_dimensions[0] = outputShape[1];
                spatial_dimensions[1] = outputShape[2];
            }
        }

        if (padding_scheme == 1) {
            auto_pad = ngraph::op::PadType::SAME_UPPER;
        } else {
            auto_pad = ngraph::op::PadType::VALID;
        }

        outputShapeNode =
            createConstNode(ngraph::element::i32, {spatial_dimensions_size}, spatial_dimensions);

        padding_left = 0;
        padding_right = 0;
        padding_top = 0;
        padding_bottom = 0;
    }

    std::shared_ptr<ngraph::Node> filterNode, biasNode;
    const auto& biasIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 2);

    filterNode = getInputNode(1);
    biasNode = getInputNode(2);

    if (checkInputOperandType(1, (int32_t)OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL)) {
        auto filterIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
        const auto& filterOperand = sModelInfo->getOperand(filterIndex);
        vec<float> filterScales = filterOperand.extraParams.channelQuant().scales;
        float inputScale = sModelInfo->getOperandScale(0);
        auto filterScalesNode =
            createConstNode(ngraph::element::f32, ngraph::Shape{filterScales.size()}, filterScales);
        auto inputScalesNode =
            createConstNode(ngraph::element::f32, ngraph::Shape{1}, convertToVector(inputScale));

        // for quant symm per channel type inputs, bias is of type TENSOR_INT32. For TENSOR_INT32
        // type, dequantization is not applied during node creation
        // bias_scale[i] = input_scale * filter_scale[i]
        auto biasScalMultiplier =
            std::make_shared<ngraph::opset3::Multiply>(filterScalesNode, inputScalesNode);
        biasNode = std::make_shared<ngraph::opset3::Convert>(biasNode, ngraph::element::f32);
        biasNode = std::make_shared<ngraph::opset3::Multiply>(biasNode, biasScalMultiplier);
    } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM) ||
               checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM_SIGNED)) {
        // for quant type inputs, bias is of type TENSOR_INT32. For TENSOR_INT32 type,
        // dequantization is not applied during node creation
        biasNode = DequantizeNode(biasNode, biasIndex, ngraph::element::f32);
    }

    // OpenVino expects filter in OIHW format
    filterNode = transpose(IHWO_OIHW, filterNode);

    if (!useNchw) {  // No conversion needed if useNchw set
        inputNode = transpose(NHWC_NCHW, inputNode);
    }

    strides = {(size_t)stride_height, (size_t)stride_width};
    pads_begin = {padding_top, padding_left};
    pads_end = {padding_bottom, padding_right};
    dilations = {(size_t)dilation_height_factor, (size_t)dilation_width_factor};

    std::shared_ptr<ngraph::Node> transposeConvNode;

    if (outputShapeNode == nullptr)
        transposeConvNode = std::make_shared<ngraph::opset3::ConvolutionBackpropData>(
            inputNode, filterNode, ngraph::Strides(strides), ngraph::CoordinateDiff(pads_begin),
            ngraph::CoordinateDiff(pads_end), ngraph::Strides(dilations));
    else
        transposeConvNode = std::make_shared<ngraph::opset3::ConvolutionBackpropData>(
            inputNode, filterNode, outputShapeNode, ngraph::Strides(strides),
            ngraph::CoordinateDiff(pads_begin), ngraph::CoordinateDiff(pads_end),
            ngraph::Strides(dilations), auto_pad);

    auto biasDimensions = getInputOperandDimensions(2);
    std::vector<uint32_t> shape(transposeConvNode->get_shape().size(), 1);
    shape[1] = biasDimensions[0];
    auto shapeNode = createConstNode(ngraph::element::i32, ngraph::Shape{shape.size()}, shape);

    biasNode = std::make_shared<ngraph::opset3::Reshape>(biasNode, shapeNode, true);

    std::shared_ptr<ngraph::Node> outputNode = std::make_shared<ngraph::opset3::Add>(
        transposeConvNode, biasNode, ngraph::op::AutoBroadcastType::NUMPY);
    outputNode = applyActivation(outputNode, activationFn);

    if (!useNchw) {
        outputNode = transpose(NCHW_NHWC, outputNode);
    }

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
