#include <Transpose.hpp>
#undef LOG_TAG
#define LOG_TAG "Transpose"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Transpose::Transpose(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Transpose::validate() {
    // TODO: Add Support for all_tensors_as_inputs
    const auto& dimsOperandIndex1 = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
    const auto inputRank = getInputOperandDimensions(0).size();
    if ( !isValidInputTensor(0) || inputRank > 4) {
         ALOGE("%s Empty  or Invalid dimensions size for input", __func__);
         return false;
    }

    if(!sModelInfo->isOperandLifeTimeConst(dimsOperandIndex1)) {
        ALOGE("%s Only Const lifetime is supported", __func__);
        return false;
    }

    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    if (inputsSize == 2) {
        const auto& dimsOperandIndex2 = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
        if (!isValidInputTensor(1) || !sModelInfo->isOperandLifeTimeConst(dimsOperandIndex2)) {
            ALOGE("%s Invalid operand type or operand lifetime", __func__);
            return false;
        }
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Transpose::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;

    input = getInputNode(0);

    std::shared_ptr<ngraph::Node> order;
    order = createConstNode(ngraph::element::i32, {0}, convertToVector(0));

    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    if (inputsSize == 2) {
        order = getInputNode(1);
    }
    std::shared_ptr<ngraph::Node> outputNode;

    outputNode = std::make_shared<ngraph::opset3::Transpose>(input, order);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
