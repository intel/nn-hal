#include <Channel_Shuffle.hpp>
#define LOG_TAG "Channel_Shuffle"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Channel_Shuffle::Channel_Shuffle(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Channel_Shuffle::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        ALOGE(
            "%s: Output operand 0 is not of type TENSOR_FLOAT32/TENSOR_QUANT8_ASYMM. Unsupported "
            "operation",
            __func__);
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        ALOGE(
            "%s: Input operand 0 is not of type TENSOR_FLOAT32/TENSOR_QUANT8_ASYMM. Unsupported "
            "operation",
            __func__);
        return false;
    }
    if (!checkInputOperandType(1, (int32_t)OperandType::INT32)) {
        ALOGE("%s: Input operand 1 is not of type I32. Unsupported operation", __func__);
        return false;
    }
    if (!checkInputOperandType(2, (int32_t)OperandType::INT32)) {
        ALOGE("%s: Input operand 2 is not of type I32. Unsupported operation", __func__);
        return false;
    }

    // Check input rank
    const int64_t inputRank = getInputOperandDimensions(0).size();
    if (inputRank > 4 || inputRank <= 0) {
        ALOGE("%s Invalid input dimensions size!", __func__);
        return false;
    }

    // Check axis range
    int64_t axis = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 2);
    if (!(axis >= -inputRank && axis < inputRank)) {
        ALOGE("%s Axis not in the range [-inputRank, inputRank)", __func__, axis);
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Channel_Shuffle::createNode() {
    // Creating input nodes
    auto inputNode = getInputNode(0);
    int64_t group = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);
    int64_t axis = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 2);

    auto inputRank = getInputOperandDimensions(0).size();
    axis = (axis >= 0) ? axis : (axis + inputRank);

    // Convert the inputNode to 4D
    std::shared_ptr<ngraph::Node> squeezeAxes;
    if (inputRank < 4) {
        switch (inputRank) {
            case 1:
                squeezeAxes = std::make_shared<ngraph::opset3::Constant>(
                    ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{1, 2, 3});
                break;
            case 2:
                squeezeAxes = std::make_shared<ngraph::opset3::Constant>(
                    ngraph::element::i64, ngraph::Shape{2}, std::vector<int64_t>{2, 3});
                break;
            case 3:
                squeezeAxes = std::make_shared<ngraph::opset3::Constant>(
                    ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{3});
                break;
            default:
                break;
        }

        inputNode = std::make_shared<ngraph::opset3::Unsqueeze>(inputNode, squeezeAxes);
    }

    std::shared_ptr<ngraph::Node> outputNode =
        std::make_shared<ngraph::opset3::ShuffleChannels>(inputNode, axis, group);

    // Using squeeze to convert the shape of outputNode to shape of inputNode before unsqueeze
    if (inputRank < 4) {
        outputNode = std::make_shared<ngraph::opset3::Squeeze>(outputNode, squeezeAxes);
    }

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
