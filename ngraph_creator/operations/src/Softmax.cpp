//#define LOG_NDEBUG 0
#include <Softmax.hpp>
#define LOG_TAG "Softmax"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Softmax::Softmax(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Softmax::validate() {
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        return false;
    }
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        return false;
    }
    if (!checkInputOperandType(1, (int32_t)OperandType::FLOAT32)) {
        return false;
    }
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Softmax::createNode() {
    auto inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
    auto inputOp = mNgraphNodes->getOperationOutput(inputIndex);

    std::shared_ptr<ngraph::Node> outputNode = std::make_shared<ngraph::opset3::Softmax>(inputOp);

    float beta = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 1);
    ALOGV("Softmax beta = %f ", beta);

    if (beta <= 0.0f) ALOGE("beta must be positive for softmax");

    const auto outputOperand = sModelInfo->getOperand(mDefaultOutputIndex);
    if (outputOperand.lifetime == OperandLifeTime::MODEL_OUTPUT)
        addResultNode(mDefaultOutputIndex, outputNode);

    return outputNode;
}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
