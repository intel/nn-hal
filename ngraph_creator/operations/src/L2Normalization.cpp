#include <L2Normalization.hpp>
#undef LOG_TAG
#define LOG_TAG "L2Normalization"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

L2Normalization::L2Normalization(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool L2Normalization::validate() {
    const auto inputRank = getInputOperandDimensions(0).size();
    if ((inputRank > 4) || (!isValidInputTensor(0))) {
        ALOGE("%s Invalid dimensions size for input(%lu)", __func__, inputRank);
        return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ov::Node> L2Normalization::createNode() {
    std::shared_ptr<ov::Node> inputNode;

    int32_t inputAxes = -1;
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    ALOGD("%s inputsSize %lu", __func__, inputsSize);
    inputNode = getInputNode(0);
    // NN-HAL 1.2 specific optional input
    if (inputsSize == 2) {
        inputAxes = sModelInfo->ParseOperationInput<int32_t>(mNnapiOperationIndex, 1);
    }
    auto inputAxesNode = createConstNode(ov::element::i32, {1}, convertToVector(inputAxes));
    // TODO: Add support for NNAPI feature level 4, if the elements along an axis are all zeros, the
    // result is undefined. Since NNAPI feature level 4, if the elements along an axis are all
    // zeros, the result is logical zero.

    /*
     *         output[batch, row, col, channel] =
     *         input[batch, row, col, channel] /
     *         sqrt(sum_{c} pow(input[batch, row, col, c], 2))
     */
    auto mul = std::make_shared<ov::opset3::Multiply>(inputNode, inputNode);
    auto sum = std::make_shared<ov::opset3::ReduceSum>(mul, inputAxesNode, true);
    auto sqrt = std::make_shared<ov::opset3::Sqrt>(sum);
    auto outputNode = std::make_shared<ov::opset3::Divide>(inputNode, sqrt);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
