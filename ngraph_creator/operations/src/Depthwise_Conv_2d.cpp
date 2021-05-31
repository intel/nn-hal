#include <Depthwise_Conv_2d.hpp>
#include <NgraphHelper.hpp>
#define LOG_TAG "Depthwise_Conv_2d"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Depthwise_Conv_2d::Depthwise_Conv_2d(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Depthwise_Conv_2d::validate() {
    // Check Output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM))
        return false;

    for (int i = 0; i < 2; i++) {
        // Check input/filter operands(0/1) are of type TENSOR_FLOAT32/TENSOR_QUANT8_ASYMM
        if (!checkInputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32) &&
            !checkInputOperandType(i, (int32_t)OperandType::TENSOR_QUANT8_ASYMM))
            return false;
    }
    // Check bias type
    if (!checkInputOperandType(2, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(2, (int32_t)OperandType::TENSOR_INT32))
        return false;

    // Check Input, Filter Dimension size
    const auto& inputDimensionsSize = getInputOperandDimensions(0).size();
    const auto& filterDimensionsSize = getInputOperandDimensions(1).size();
    if (inputDimensionsSize != 4 || filterDimensionsSize != 4) {
        ALOGE("%s Invalid dimensions size for input(%d) or filter(%d)", __func__,
              inputDimensionsSize, filterDimensionsSize);
        return false;
    }

    const auto& filterDimensions = getInputOperandDimensions(1);
    if (filterDimensions[0] != 1) {
        ALOGE("%s Invalid dimension at filter[0] (%d)", __func__, filterDimensions[0]);
        return false;
    }

    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);

    if (inputsSize >= 11 && inputsSize <= 14 &&
        !checkInputOperandType(8, (int32_t)OperandType::BOOL)) {
        // Checking input types for explicit padding
        for (int i = 3; i < 11; i++) {
            if (!checkInputOperandType(i, (int32_t)OperandType::INT32)) return false;
        }

        if (inputsSize > 11 && inputsSize <= 14) {
            switch (inputsSize) {
                case 14:
                    if (!checkInputOperandType(13, (int32_t)OperandType::INT32)) return false;
                    ;
                case 13:
                    if (!checkInputOperandType(12, (int32_t)OperandType::INT32)) return false;
                case 12:
                    if (!checkInputOperandType(11, (int32_t)OperandType::BOOL)) return false;
                default:
                    break;
            }
        }
    } else if (inputsSize >= 8 && inputsSize <= 11) {
        // Checking input types for implicit padding
        for (int i = 3; i < 8; i++) {
            if (!checkInputOperandType(i, (int32_t)OperandType::INT32)) return false;
        }

        if (inputsSize > 8 && inputsSize <= 11) {
            switch (inputsSize) {
                case 11:
                    if (!checkInputOperandType(10, (int32_t)OperandType::INT32)) return false;
                case 10:
                    if (!checkInputOperandType(9, (int32_t)OperandType::INT32)) return false;
                case 9:
                    if (!checkInputOperandType(8, (int32_t)OperandType::BOOL)) return false;
                default:
                    break;
            }
        }
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Depthwise_Conv_2d::createNode() {
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    ALOGD("%s inputsSize %d", __func__, inputsSize);
    bool isImplicit = false, isExplicit = false;

    if (inputsSize >= 11 && inputsSize <= 14 &&
        !checkInputOperandType(8, (int32_t)OperandType::BOOL)) {
        isExplicit = true;
    } else if (inputsSize >= 8 && inputsSize <= 11) {
        isImplicit = true;
    }

    int32_t padding_left, padding_right;
    int32_t padding_top, padding_bottom;
    int32_t stride_width, stride_height;
    int32_t dilation_width_factor = 1, dilation_height_factor = 1;
    int32_t depthwise_multiplier;
    int32_t activationFn;
    int32_t layout;
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

        depthwise_multiplier = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 9);

        activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 10);

        if (inputsSize > 11 && inputsSize <= 14) {
            switch (inputsSize) {
                case 14:
                    dilation_height_factor =
                        sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 13);
                case 13:
                    dilation_width_factor =
                        sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 12);
                case 12:
                    layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 11);
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
                input_channel = inputDimensions[1];
            } else {
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

        depthwise_multiplier = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 6);

        activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 7);

        if (inputsSize > 8 && inputsSize <= 11) {
            switch (inputsSize) {
                case 11:
                    dilation_height_factor =
                        sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 10);
                case 10:
                    dilation_width_factor =
                        sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 9);
                case 9:
                    layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 8);
                default:
                    break;
            }
        }

        if (layout) useNchw = true;

        {
            if (useNchw) {
                input_width = inputDimensions[3];
                input_height = inputDimensions[2];
                input_channel = inputDimensions[1];
            } else {
                input_width = inputDimensions[2];
                input_height = inputDimensions[1];
                input_channel = inputDimensions[3];
            }
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

    std::shared_ptr<ngraph::Node> inputNode, filterNode, biasNode;

    const auto& inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
    const auto& filterIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
    const auto& biasIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 2);

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        inputNode = getInputNode<float>(0);
        filterNode = getInputNode<float>(1);
        biasNode = getInputNode<float>(2);
    } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        inputNode = getInputNode<uint8_t>(0);
        filterNode = getInputNode<uint8_t>(1);
        biasNode = getInputNode<int>(2);

        inputNode = DequantizeNode(inputNode, inputIndex, ngraph::element::f32);
        filterNode = DequantizeNode(filterNode, filterIndex, ngraph::element::f32);
        biasNode = DequantizeNode(biasNode, biasIndex, ngraph::element::f32);
    }
    // OpenVino expects filter in OIHW format
    filterNode = transpose(IHWO_OIHW, filterNode);
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

    ALOGD("input_channel value is ******* %d", input_channel);

    if (filterNode != nullptr) {
        std::vector<size_t> shape(&filterNode->get_shape()[0], &filterNode->get_shape()[0] + 4);
        shape[0] /= input_channel;
        shape.insert(shape.begin(), input_channel);
        ALOGD("%s final filternode shape %d", __func__, shape.size());

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
        mNgraphNodes->setForcedNchw(mDefaultOutputIndex, false);
    }

    if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        const auto& outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
        outputNode = QuantizeNode(outputNode, outputIndex, ngraph::element::u8);
    }

    const auto outputLifetime = sModelInfo->getOperandLifetime(mDefaultOutputIndex);
    if (outputLifetime == OperandLifeTime::MODEL_OUTPUT) {
        addResultNode(mDefaultOutputIndex, outputNode);
    }
    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
