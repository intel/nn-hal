#include <L2_Normalization.hpp>
#include <NgraphHelper.hpp>
#define LOG_TAG "L2_Normalization"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

L2_Normalization::L2_Normalization(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool L2_Normalization::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }
    const auto inputRank = getInputOperandDimensions(0).size();
    if ((inputRank > 4) || (!isValidInputTensor(0))) {
        ALOGE("%s Invalid dimensions size for input(%d)", __func__, inputRank);
        return false;
    }
    if (inputsSize == 2) {
        if (!checkInputOperandType(1, (int32_t)OperandType::INT32)) {
            return false;
        }
        auto inputAxes = sModelInfo->ParseOperationInput<int32_t>(mNnapiOperationIndex, 1);
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> L2_Normalization::createNode() {
    std::shared_ptr<ngraph::Node> inputNode;
    float eps = 1e-6f;
    auto epsMode = ngraph::op::EpsMode::MAX;
    const auto inputRank = getInputOperandDimensions(0).size();
    int32_t inputAxes = -1;
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    ALOGD("%s inputsSize %d", __func__, inputsSize);
    inputNode = getInputNode(0);
    if (inputsSize == 2) {
        inputAxes = sModelInfo->ParseOperationInput<int32_t>(mNnapiOperationIndex, 1);
    }
    auto inputAxesNode = createConstNode(ngraph::element::i32, {1}, convertToVector(inputAxes));
    /*
     *         output[batch, row, col, channel] =
     *         input[batch, row, col, channel] /
     *         sqrt(sum_{c} pow(input[batch, row, col, c], 2))
     */
    auto mul = std::make_shared<ngraph::opset3::Multiply>(inputNode, inputNode);
    auto sum = std::make_shared<ngraph::opset3::ReduceSum>(mul, inputAxesNode, true);
    auto sqrt = std::make_shared<ngraph::opset3::Sqrt>(sum);
    auto outputNode = std::make_shared<ngraph::opset3::Divide>(inputNode, sqrt);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
