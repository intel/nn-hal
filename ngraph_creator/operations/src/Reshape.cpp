#include <Reshape.hpp>
#undef LOG_TAG
#define LOG_TAG "Reshape"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Reshape::Reshape(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Reshape::validate() {
    const auto& dimsOperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
    if (!sModelInfo->isOperandLifeTimeConst(dimsOperandIndex) || !isValidInputTensor(1)) {
        // TODO: Support CPU_reshape_all_tensors_as_inputs
        ALOGE("%s Only Constant non-zero dimensions supported now", __func__);
        return false;
    }
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ov::Node> Reshape::createNode() {
    const auto& dimsOperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
    auto outDims = sModelInfo->GetConstVecOperand<int32_t>(dimsOperandIndex);
    VLOGDIMS(L3, outDims, "Reshape::createNode dims");
    std::shared_ptr<ov::Node> inputOp;
    inputOp = getInputNode(0);

    const auto& inDims = getInputOperandDimensions(0);
    auto numInputElements = 1;
    int strechDim = -1;
    auto numOutputElements = 1;

    for (size_t i = 0; i < inDims.size(); i++) numInputElements *= inDims[i];

    for (size_t i = 0; i < outDims.size(); i++) {
        if ((int)outDims[i] < 0) {
            strechDim = i;
            continue;
        }
        numOutputElements *= outDims[i];
    }
    if (strechDim >= 0) {
        auto strechValue = numInputElements / numOutputElements;
        outDims[strechDim] = (uint32_t)strechValue;
        numOutputElements *= strechValue;

        VLOGDIMS(L3, outDims, "Reshape::outDims with stretch dimension introduced");
    }

    if (numInputElements != numOutputElements) {
        ALOGE("numInputElements = %d is not equal to numOutputElements = %d", numInputElements,
              numOutputElements);
    }

    auto shapeNode = std::make_shared<ov::opset3::Constant>(
        ov::element::i32, ov::Shape{outDims.size()}, outDims.data());

    std::shared_ptr<ov::Node> outputNode =
        std::make_shared<ov::opset3::Reshape>(inputOp, shapeNode, true);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
