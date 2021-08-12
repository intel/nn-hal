#include <Dequantize.hpp>
#define LOG_TAG "Dequantize"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Dequantize::Dequantize(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Dequantize::validate() {
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Dequantize::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;

    input = getInputNode(0, false);

    const auto& inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);

    std::shared_ptr<ngraph::Node> outputNode;

    // TODO: create a generic function for all TENSOR_QUANT8_SYMM_PER_CHANNEL tensors
    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL)) {
        const auto& inputOperand = sModelInfo->getOperand(inputIndex);
        vec<float> inputScales = inputOperand.extraParams.channelQuant().scales;
        auto channelDim = inputOperand.extraParams.channelQuant().channelDim;

        std::shared_ptr<ngraph::Node> inputScales_node;

        if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT16)) {
            input = std::make_shared<ngraph::opset3::Convert>(input, ngraph::element::f16);
            inputScales_node = createConstNode(ngraph::element::f16,
                                               ngraph::Shape{inputScales.size()}, inputScales);
        } else {
            input = std::make_shared<ngraph::opset3::Convert>(input, ngraph::element::f32);
            inputScales_node = createConstNode(ngraph::element::f32,
                                               ngraph::Shape{inputScales.size()}, inputScales);
        }

        const auto inputRank = getInputOperandDimensions(0).size();
        if (channelDim == (inputRank - 1)) {
            outputNode = std::make_shared<ngraph::opset3::Multiply>(input, inputScales_node);
        } else {
            if (inputRank == 4) {
                switch (channelDim) {
                    case 2:
                        input = transpose(NHCW_NHWC, input);
                        outputNode =
                            std::make_shared<ngraph::opset3::Multiply>(input, inputScales_node);
                        outputNode = transpose(NHCW_NHWC, outputNode);
                        break;
                    case 1:
                        input = transpose(NCHW_NHWC, input);
                        outputNode =
                            std::make_shared<ngraph::opset3::Multiply>(input, inputScales_node);
                        outputNode = transpose(NHWC_NCHW, outputNode);
                        break;
                    case 0:
                        input = transpose(NHWC_CWHN, input);
                        outputNode =
                            std::make_shared<ngraph::opset3::Multiply>(input, inputScales_node);
                        outputNode = transpose(CWHN_NHWC, outputNode);
                        break;
                    default:
                        break;
                }
            } else if (inputRank == 3) {
                switch (channelDim) {
                    case 1:
                        input = transpose(NHC_NCH, input);
                        outputNode =
                            std::make_shared<ngraph::opset3::Multiply>(input, inputScales_node);
                        outputNode = transpose(NHC_NCH, outputNode);
                        break;
                    case 0:
                        input = transpose(CNH_NHC, input);
                        outputNode =
                            std::make_shared<ngraph::opset3::Multiply>(input, inputScales_node);
                        outputNode = transpose(NHC_CNH, outputNode);
                        break;
                    default:
                        break;
                }
            } else if (inputRank == 2) {
                switch (channelDim) {
                    case 0:
                        input = transpose(NC_CN, input);
                        outputNode =
                            std::make_shared<ngraph::opset3::Multiply>(input, inputScales_node);
                        outputNode = transpose(NC_CN, outputNode);
                        break;
                    default:
                        break;
                }
            }
        }
    } else {
        if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT16))
            outputNode = DequantizeNode(input, inputIndex, ngraph::element::f16);

        else
            outputNode = DequantizeNode(input, inputIndex, ngraph::element::f32);
    }

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
