//#define LOG_NDEBUG 0
#include <Reshape.hpp>
#define LOG_TAG "Reshape"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Reshape::Reshape(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Reshape::validate() {
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }
    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_INT32)) {
        return false;
    }
    const auto& dimsOperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
    if (!sModelInfo->isOperandLifeTimeConst(dimsOperandIndex)) {
        // TODO: Support CPU_reshape_all_tensors_as_inputs
        ALOGE("%s Only Constant dimensions supported now", __func__);
        return false;
    }
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Reshape::createNode() {
    const auto& dimsOperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
    auto outDims = sModelInfo->GetConstVecOperand<int32_t>(dimsOperandIndex);
    VLOGDIMS(L3, outDims, "Reshape::createNode dims");

    auto inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
    std::shared_ptr<ngraph::Node> inputOp;

    inputOp = getInputNode(0);

    const auto& inDims = getInputOperandDimensions(0);
    auto numInputElements = 1;
    int strechDim = -1;
    auto numOutputElements = 1;

    for (auto i = 0; i < inDims.size(); i++) numInputElements *= inDims[i];

    for (auto i = 0; i < outDims.size(); i++) {
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

    if (mNgraphNodes->isForcedNchw(inputIndex)) {
        inputOp = transpose(NCHW_NHWC, inputOp);
        mNgraphNodes->setForcedNchw(mDefaultOutputIndex, false);
    }

    auto shapeNode = std::make_shared<ngraph::opset3::Constant>(
        ngraph::element::i32, ngraph::Shape{outDims.size()}, outDims.data());

    std::shared_ptr<ngraph::Node> outputNode =
        std::make_shared<ngraph::opset3::Reshape>(inputOp, shapeNode, true);

    return outputNode;
}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
