#include <AveragePool2D.hpp>
#undef LOG_TAG
#define LOG_TAG "AveragePool2D"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

AveragePool2D::AveragePool2D(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool AveragePool2D::validate() {
    // Check Input Dimension size
    const auto& inputDimensionsSize = getInputOperandDimensions(0).size();
    if (inputDimensionsSize != 4) {
        ALOGE("%s Invalid dimensions size for input(%lu)", __func__, inputDimensionsSize);
        return false;
    }
    //check Input are of valid dimension or not
    if ( !isValidInputTensor(0)) {
         ALOGE("%s Empty  or Invalid dimensions size for input", __func__);
         return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> AveragePool2D::createNode() {
    std::shared_ptr<ngraph::Node> inputNode;
    const auto& inDims = getInputOperandDimensions(0);
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);

    inputNode = getInputNode(0);

    ALOGD("%s inputsSize %lu", __func__, inputsSize);

    bool isImplicit = false, isExplicit = false;

    int32_t layout = 0;
    bool useNchw = false;
    int32_t padding_scheme;
    std::vector<size_t> pad_begin;
    std::vector<size_t> pad_end;
    std::vector<size_t> strides;
    std::vector<size_t> kernel;
    int32_t padding_left, padding_right;
    int32_t padding_top, padding_bottom;
    int32_t stride_width, stride_height;
    int32_t filter_width, filter_height;
    int32_t input_width, input_height;
    int32_t activationFn;
    ngraph::op::PadType auto_pad;

    if (inputsSize >= 10 && inputsSize <= 11) {
        isExplicit = true;
    } else if (inputsSize >= 7 && inputsSize <= 8) {
        isImplicit = true;
    }

    if (isExplicit) {
        padding_left = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 1);
        padding_right = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);
        padding_top = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 3);
        padding_bottom = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 4);

        stride_width = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 5);
        stride_height = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 6);

        filter_width = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 7);
        filter_height = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 8);

        activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 9);

        if (inputsSize == 11) {
            layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 10);
        }

        if (layout) useNchw = true;

        auto_pad = ngraph::op::PadType::EXPLICIT;
    }

    if (isImplicit) {
        padding_scheme = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 1);

        stride_width = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);
        stride_height = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 3);

        filter_width = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 4);
        filter_height = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 5);

        activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 6);

        if (inputsSize == 8) {
            layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 7);
        }

        if (layout) useNchw = true;

        if (useNchw) {
            input_width = inDims[3];
            input_height = inDims[2];
        } else {
            input_width = inDims[2];
            input_height = inDims[1];
        }

        if (padding_scheme == 1) {
            calculateExplicitPadding(input_width, stride_width, filter_width, 1, &padding_left,
                                     &padding_right);
            calculateExplicitPadding(input_height, stride_height, filter_height, 1, &padding_top,
                                     &padding_bottom);

            auto_pad = ngraph::op::PadType::SAME_UPPER;

        } else {
            padding_left = 0;
            padding_right = 0;
            padding_top = 0;
            padding_bottom = 0;
            auto_pad = ngraph::op::PadType::VALID;
        }
    }

    if (!useNchw) {  // No conversion needed if useNchw set
        inputNode = transpose(NHWC_NCHW, inputNode);
    }

    strides = {(size_t)stride_height, (size_t)stride_width};
    pad_begin = {(size_t)padding_top, (size_t)padding_left};
    pad_end = {(size_t)padding_bottom, (size_t)padding_right};
    kernel = {(size_t)filter_height, (size_t)filter_width};

    std::shared_ptr<ngraph::Node> outputNode = std::make_shared<ngraph::opset3::AvgPool>(
        inputNode, ngraph::Strides(strides), ngraph::Shape(pad_begin), ngraph::Shape(pad_end),
        ngraph::Shape(kernel), true, ngraph::op::RoundingType::FLOOR, auto_pad);

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
