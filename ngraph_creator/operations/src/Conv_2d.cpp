#include <Conv_2d.hpp>
#include <NgraphHelper.hpp>
#define LOG_TAG "Conv_2d"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Conv_2d::Conv_2d(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Conv_2d::validate() {
    // Check Output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) return false;

    for (int i = 0; i <= 2; i++) {
        // Check input/filter/bias operands(0/1/2) are of type TENSOR_FLOAT32
        if (!checkInputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }

    // Check Input, Filter Dimension size
    const auto& inputDimensionsSize = getInputOperandDimensions(0).size();
    const auto& filterDimensionsSize = getInputOperandDimensions(1).size();
    if (inputDimensionsSize != 4 || filterDimensionsSize != 4) {
        ALOGE("%s Invalid dimensions size for input(%d) or filter(%d)", __func__,
              inputDimensionsSize, filterDimensionsSize);
        return false;
    }

    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    if (inputsSize >= 10 && checkInputOperandType(7, (int32_t)OperandType::INT32)) {
        // Explicit Padding if 10 or more inputs present and index 7 is INT32.
        // Check all other Input operand types for explicit Padding
        for (int i = 3; i < inputsSize; i++) {
            // All inputs except index 10 should be INT32
            // index 10 should be BOOL
            if (i == 10) {
                if (!checkInputOperandType(i, (int32_t)OperandType::BOOL)) return false;
            } else if (!checkInputOperandType(i, (int32_t)OperandType::INT32))
                return false;
        }
    } else if (inputsSize >= 7 && inputsSize <= 10) {
        // Check all other Input operand types for implicit Padding
        for (int i = 3; i < inputsSize; i++) {
            // All inputs except index 7 should be INT32
            // index 7 should be BOOL
            if (i == 7) {
                if (!checkInputOperandType(i, (int32_t)OperandType::BOOL)) return false;
            } else if (!checkInputOperandType(i, (int32_t)OperandType::INT32))
                return false;
        }
    } else {
        ALOGE("%s FAILED inputsSize %d", __func__);
        return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Conv_2d::createNode() {
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    ALOGD("%s inputsSize %d", __func__, inputsSize);

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

    {
        const auto& inputDimensions = getInputOperandDimensions(0);
        input_width = inputDimensions[2];
        input_height = inputDimensions[1];
    }
    {
        const auto& filterDimensions = getInputOperandDimensions(1);
        filter_width = filterDimensions[2];
        filter_height = filterDimensions[1];
    }

    if (inputsSize >= 10 && checkInputOperandType(7, (int32_t)OperandType::INT32)) {
        // Explicit Padding if 10 or more inputs present and index 7 is INT32.
        padding_left = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 3);
        padding_right = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 4);
        padding_top = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 5);
        padding_bottom = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 6);

        stride_width = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 7);
        stride_height = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 8);

        activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 9);

        switch (inputsSize) {
            case 13:
                dilation_height_factor =
                    sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 12);
            case 12:
                dilation_width_factor =
                    sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 11);
            case 11:
                layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 10);
            default:
                break;
        }

        if (layout) useNchw = true;

        auto_pad = ngraph::op::PadType::EXPLICIT;
    } else if (inputsSize == 7 ||
               (inputsSize <= 10 && checkInputOperandType(7, (int32_t)OperandType::BOOL))) {
        // Implicit Padding if 7 to 10 inputs present and index 7 is BOOL.
        padding_scheme = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 3);

        stride_width = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 4);
        stride_height = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 5);

        activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 6);
        switch (inputsSize) {
            case 10:
                dilation_height_factor =
                    sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 9);
            case 9:
                dilation_width_factor =
                    sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 8);
            case 8:
                layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 7);
            default:
                break;
        }

        if (layout) useNchw = true;

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
            auto_pad = ngraph::op::PadType::NOTSET;
        }
    } else {
        ALOGE("%s FAILED inputsSize %d not defined", __func__);
    }

    auto inputNode = getInputNode<float>(0);
    auto filterNode = getInputNode<float>(1);
    // OpenVino expects filter in OIHW format
    filterNode = transpose(OHWI_OIHW, filterNode);
    auto inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
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

    strides = {(size_t)stride_width, (size_t)stride_height};
    pads_begin = {padding_left, padding_top};
    pads_end = {padding_right, padding_bottom};
    dilations = {(size_t)dilation_width_factor, (size_t)dilation_height_factor};
    auto convNode = std::make_shared<ngraph::opset3::Convolution>(
        inputNode, filterNode, ngraph::Strides(strides), ngraph::CoordinateDiff(pads_begin),
        ngraph::CoordinateDiff(pads_end), ngraph::Strides(dilations), auto_pad);

    auto biasNode = getInputNode<float>(2);
    auto biasDimensions = getInputOperandDimensions(2);
    std::vector<uint32_t> shape(convNode->get_shape().size(), 1);
    shape[1] = biasDimensions[0];
    auto shapeNode = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i32,
                                                                ngraph::Shape{shape.size()}, shape);
    biasNode = std::make_shared<ngraph::opset3::Reshape>(biasNode, shapeNode, true);

    std::shared_ptr<ngraph::Node> outputNode = std::make_shared<ngraph::opset3::Add>(
        convNode, biasNode, ngraph::op::AutoBroadcastType::NUMPY);
    outputNode = applyActivation(outputNode, activationFn);

    const auto outputLifetime = sModelInfo->getOperandLifetime(mDefaultOutputIndex);
    if (outputLifetime == OperandLifeTime::MODEL_OUTPUT) {
        outputNode = transpose(NCHW_NHWC, outputNode);
        mNgraphNodes->setForcedNchw(mDefaultOutputIndex, false);
        addResultNode(mDefaultOutputIndex, outputNode);
    }
    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
