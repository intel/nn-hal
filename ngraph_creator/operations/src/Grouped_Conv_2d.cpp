#include <Grouped_Conv_2d.hpp>
#include <NgraphHelper.hpp>
#define LOG_TAG "Grouped_Conv_2d"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Grouped_Conv_2d::Grouped_Conv_2d(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Grouped_Conv_2d::validate() {
    // Check Output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM))
        return false;

    // Check input/filter operands(0/1) are of type TENSOR_FLOAT32/TENSOR_QUANT8_ASYMM
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM))
        return false;
    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(1, (int32_t)OperandType::TENSOR_QUANT8_ASYMM) &&
        !checkInputOperandType(1, (int32_t)OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL))
        return false;
    // Check bias type
    if (!checkInputOperandType(2, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(2, (int32_t)OperandType::TENSOR_INT32))
        return false;

    // Check Input, Filter Dimension size
    const auto& inputDimensionsSize = getInputOperandDimensions(0).size();
    const auto& filterDimensionsSize = getInputOperandDimensions(1).size();
    if (inputDimensionsSize != 4 || filterDimensionsSize != 4) {
        ALOGE("%s Invalid dimensions size for input(%lu) or filter(%lu)", __func__,
              inputDimensionsSize, filterDimensionsSize);
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

std::shared_ptr<ngraph::Node> Grouped_Conv_2d::createNode() {
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    bool isImplicit = false, isExplicit = false;

    if (inputsSize >= 11 && inputsSize <= 12 &&
        !checkInputOperandType(8, (int32_t)OperandType::BOOL)) {
        isExplicit = true;
    } else if (inputsSize >= 8 && inputsSize <= 9) {
        isImplicit = true;
    }

    int32_t padding_left, padding_right;
    int32_t padding_top, padding_bottom;
    int32_t stride_width, stride_height;
    int32_t dilation_width_factor = 1, dilation_height_factor = 1;
    int32_t number_groups;
    int32_t activationFn;
    int32_t layout = 0;
    int32_t padding_scheme;
    int32_t input_width, input_height, input_channel;
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

    if (isExplicit) {
        padding_left = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 3);
        padding_right = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 4);
        padding_top = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 5);
        padding_bottom = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 6);

        stride_width = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 7);
        stride_height = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 8);

        number_groups = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 9);

        activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 10);

        if (inputsSize == 12)
            layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 11);

        if (layout) useNchw = true;

        auto_pad = ngraph::op::PadType::EXPLICIT;
        {
            if (useNchw) {  // NCHW
                input_width = inputDimensions[3];
                input_height = inputDimensions[2];
                input_channel = inputDimensions[1];
            } else {  // NHWC
                input_width = inputDimensions[2];
                input_height = inputDimensions[1];
                input_channel = inputDimensions[3];
            }
        }
    }

    if (isImplicit) {
        padding_scheme = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 3);

        stride_width = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 4);
        stride_height = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 5);

        number_groups = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 6);

        activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 7);

        if (inputsSize == 9)
            layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 8);

        if (layout) useNchw = true;

        if (useNchw) {
            input_width = inputDimensions[3];
            input_height = inputDimensions[2];
            input_channel = inputDimensions[1];
        } else {
            input_width = inputDimensions[2];
            input_height = inputDimensions[1];
            input_channel = inputDimensions[3];
        }

        if (padding_scheme == 1) {
            calculateExplicitPadding(input_width, stride_width, filter_width, 1, &padding_left,
                                     &padding_right);
            calculateExplicitPadding(input_height, stride_height, filter_height, 1, &padding_top,
                                     &padding_bottom);
            auto_pad = ngraph::op::PadType::SAME_UPPER;
        } else if (padding_scheme == 2) {
            auto_pad = ngraph::op::PadType::VALID;
            padding_left = 0;
            padding_right = 0;
            padding_top = 0;
            padding_bottom = 0;
        } else {
            auto_pad = ngraph::op::PadType::VALID;
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
        auto filterScales_node =
            createConstNode(ngraph::element::f32, ngraph::Shape{filterScales.size()}, filterScales);
        auto inputScales_node =
            createConstNode(ngraph::element::f32, ngraph::Shape{1}, convertToVector(inputScale));
        filterNode = transpose(NHWC_CWHN, filterNode);
        auto filterNodeConvert =
            std::make_shared<ngraph::opset3::Convert>(filterNode, ngraph::element::f32);
        auto filterNodeMulScales =
            std::make_shared<ngraph::opset3::Multiply>(filterNodeConvert, filterScales_node);
        filterNode = transpose(CWHN_NHWC, filterNodeMulScales);
        // for quant symm per channel type inputs, bias is of type TENSOR_INT32. For TENSOR_INT32
        // type, dequantization is not applied during node creation
        // bias_scale[i] = input_scale * filter_scale[i]
        auto biasScalMultiplier =
            std::make_shared<ngraph::opset3::Multiply>(filterScales_node, inputScales_node);
        biasNode = std::make_shared<ngraph::opset3::Convert>(biasNode, ngraph::element::f32);
        biasNode = std::make_shared<ngraph::opset3::Multiply>(biasNode, biasScalMultiplier);
    } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        // for quant type inputs, bias is of type TENSOR_INT32. For TENSOR_INT32 type,
        // dequantization is not applied during node creation
        biasNode = DequantizeNode(biasNode, biasIndex, ngraph::element::f32);
    }

    // OpenVino expects filter in OIHW format
    filterNode = transpose(OHWI_OIHW, filterNode);

    if (!useNchw) {  // No conversion needed if useNchw set
        inputNode = transpose(NHWC_NCHW, inputNode);
    }

    strides = {(size_t)stride_width, (size_t)stride_height};
    pads_begin = {padding_left, padding_top};
    pads_end = {padding_right, padding_bottom};
    dilations = {(size_t)dilation_width_factor, (size_t)dilation_height_factor};

    // input = [batches, height, width, depth_in=(c_in * groups)]
    // filter = [depth_out=(groups * c_out), height, width, depth_group=c_in]
    // openvino: input=[n, groups * c_in, height, width], kernel=[groups, c_out,
    // c_in, height, width]
    if (filterNode != nullptr) {
        std::vector<size_t> shape(&filterNode->get_shape()[0], &filterNode->get_shape()[0] + 4);

        shape[0] /= number_groups;
        shape.insert(shape.begin(), number_groups);
        ALOGD("%s final filternode shape %lu", __func__, shape.size());

        auto shapeNode = std::make_shared<ngraph::op::Constant>(
            ngraph::element::i64, ngraph::Shape{shape.size()}, shape.data());
        filterNode = std::make_shared<ngraph::op::v1::Reshape>(filterNode, shapeNode, true);
    }

    auto groupConvNode = std::make_shared<ngraph::opset3::GroupConvolution>(
        inputNode, filterNode, ngraph::Strides(strides), ngraph::CoordinateDiff(pads_begin),
        ngraph::CoordinateDiff(pads_end), ngraph::Strides(dilations), auto_pad);

    auto biasDimensions = getInputOperandDimensions(2);
    std::vector<uint32_t> shape(groupConvNode->get_shape().size(), 1);
    shape[1] = biasDimensions[0];
    auto shapeNode = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i32,
                                                                ngraph::Shape{shape.size()}, shape);
    biasNode = std::make_shared<ngraph::opset3::Reshape>(biasNode, shapeNode, true);

    std::shared_ptr<ngraph::Node> outputNode = std::make_shared<ngraph::opset3::Add>(
        groupConvNode, biasNode, ngraph::op::AutoBroadcastType::NUMPY);
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
