#include <Conv2d.hpp>
#include <NgraphHelper.hpp>
#undef LOG_TAG
#define LOG_TAG "Conv2d"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Conv2d::Conv2d(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Conv2d::validate() {
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

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Conv2d::createNode() {
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    ALOGD("%s inputsSize %lu", __func__, inputsSize);

    bool isImplicit = false, isExplicit = false;

    if (inputsSize >= 10 && inputsSize <= 13 &&
        !checkInputOperandType(7, (int32_t)OperandType::BOOL)) {
        isExplicit = true;
    } else if (inputsSize >= 7 && inputsSize <= 10) {
        isImplicit = true;
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

    const auto& inputDimensions = getInputOperandDimensions(0);

    {
        const auto& filterDimensions = getInputOperandDimensions(1);
        filter_width = filterDimensions[2];
        filter_height = filterDimensions[1];
    }
    // TODO: create a generic function to support conv, depthwise, grouped_conv and transpose_conv
    // operations
    if (isExplicit) {
        padding_left = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 3);
        padding_right = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 4);
        padding_top = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 5);
        padding_bottom = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 6);

        stride_width = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 7);
        stride_height = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 8);

        activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 9);

        if (inputsSize > 10 && inputsSize <= 13) {
            switch (inputsSize) {
                case 13:
                    dilation_height_factor =
                        sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 12);
                    __attribute__((fallthrough));
                case 12:
                    dilation_width_factor =
                        sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 11);
                    __attribute__((fallthrough));
                case 11:
                    layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 10);
                    __attribute__((fallthrough));
                default:
                    break;
            }
        }

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
        padding_scheme = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 3);

        stride_width = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 4);
        stride_height = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 5);

        activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 6);

        if (inputsSize > 7 && inputsSize <= 10) {
            switch (inputsSize) {
                case 10:
                    dilation_height_factor =
                        sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 9);
                    __attribute__((fallthrough));
                case 9:
                    dilation_width_factor =
                        sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 8);
                    __attribute__((fallthrough));
                case 8:
                    layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 7);
                    __attribute__((fallthrough));
                default:
                    break;
            }
        }

        if (layout) useNchw = true;

        {
            if (useNchw) {
                input_width = inputDimensions[3];
                input_height = inputDimensions[2];
            } else {
                input_width = inputDimensions[2];
                input_height = inputDimensions[1];
            }
        }

        if (padding_scheme == 1) {
            calculateExplicitPadding(input_width, stride_width, filter_width, 1, &padding_left,
                                     &padding_right);
            calculateExplicitPadding(input_height, stride_height, filter_height, 1, &padding_top,
                                     &padding_bottom);
            auto_pad = ngraph::op::PadType::SAME_UPPER;
        } else {
            auto_pad = ngraph::op::PadType::VALID;
            padding_left = 0;
            padding_right = 0;
            padding_top = 0;
            padding_bottom = 0;
        }
    }

    std::shared_ptr<ngraph::Node> inputNode, filterNode, biasNode;
    const auto& biasIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 2);

    inputNode = getInputNode(0);
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
    filterNode = transpose(OHWI_OIHW, filterNode);
    if (!useNchw) {  // No conversion needed if useNchw set
        inputNode = transpose(NHWC_NCHW, inputNode);
    }

    strides = {(size_t)stride_height, (size_t)stride_width};
    pads_begin = {padding_top, padding_left};
    pads_end = {padding_bottom, padding_right};
    dilations = {(size_t)dilation_height_factor, (size_t)dilation_width_factor};

    auto convNode = std::make_shared<ngraph::opset3::Convolution>(
        inputNode, filterNode, ngraph::Strides(strides), ngraph::CoordinateDiff(pads_begin),
        ngraph::CoordinateDiff(pads_end), ngraph::Strides(dilations), auto_pad);

    auto biasDimensions = getInputOperandDimensions(2);
    std::vector<uint32_t> shape(convNode->get_shape().size(), 1);
    shape[1] = biasDimensions[0];
    auto shapeNode = createConstNode(ngraph::element::i32, ngraph::Shape{shape.size()}, shape);

    biasNode = std::make_shared<ngraph::opset3::Reshape>(biasNode, shapeNode, true);

    std::shared_ptr<ngraph::Node> outputNode = std::make_shared<ngraph::opset3::Add>(
        convNode, biasNode, ngraph::op::AutoBroadcastType::NUMPY);
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
