#include <L2Pooling2D.hpp>
#undef LOG_TAG
#define LOG_TAG "L2Pooling2D"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

L2Pooling2D::L2Pooling2D(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ngraph::Node> L2Pooling2D::createNode() {
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    bool isImplicit = false, isExplicit = false;

    if (inputsSize >= 10 && inputsSize <= 11) {
        isExplicit = true;
    } else if (inputsSize >= 7 && inputsSize <= 8) {
        isImplicit = true;
    }

    int32_t padding_left, padding_right;
    int32_t padding_top, padding_bottom;
    int32_t stride_width, stride_height;
    int32_t activationFn;
    int32_t layout = 0;
    int32_t padding_scheme;
    int32_t input_width, input_height;
    int32_t filter_width, filter_height;
    bool useNchw = false;
    std::vector<size_t> strides;
    std::vector<size_t> pads_begin;
    std::vector<size_t> pads_end;
    std::vector<size_t> kernel;
    ngraph::op::PadType auto_pad;

    const auto& inputDimensions = getInputOperandDimensions(0);

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
        if (useNchw) {
            input_width = inputDimensions[3];
            input_height = inputDimensions[2];
        } else {
            input_width = inputDimensions[2];
            input_height = inputDimensions[1];
        }
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
            input_width = inputDimensions[3];
            input_height = inputDimensions[2];
        } else {
            input_width = inputDimensions[2];
            input_height = inputDimensions[1];
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
            auto_pad = ngraph::op::PadType::NOTSET;
        }
    }

    std::shared_ptr<ngraph::Node> inputNode, inputSquared, sqrtOutput;
    inputNode = getInputNode(0);
    inputSquared = std::make_shared<ngraph::op::v1::Multiply>(inputNode, inputNode);

    if (!useNchw) {
        ALOGD("%s Forced NCHW conversion at operationIndex %d", __func__, mNnapiOperationIndex);
        inputSquared = transpose(NHWC_NCHW, inputSquared);
    }

    strides = {(size_t)stride_height, (size_t)stride_width};
    kernel = {(size_t)filter_height, (size_t)filter_width};
    pads_begin = {(size_t)padding_top, (size_t)padding_left};
    pads_end = {(size_t)padding_bottom, (size_t)padding_right};

    auto avgPoolNode = std::make_shared<ngraph::op::v1::AvgPool>(
        inputSquared, ngraph::Strides(strides), ngraph::Shape(pads_begin), ngraph::Shape(pads_end),
        ngraph::Shape(kernel), true, ngraph::op::RoundingType::FLOOR, auto_pad);

    sqrtOutput = std::make_shared<ngraph::op::v0::Sqrt>(avgPoolNode);

    auto outputNode = applyActivation(sqrtOutput, activationFn);

    if (!useNchw) {
        outputNode = transpose(NCHW_NHWC, outputNode);
    }

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
