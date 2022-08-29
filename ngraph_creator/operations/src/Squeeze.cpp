#include <Squeeze.hpp>
#undef LOG_TAG
#define LOG_TAG "Squeeze"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Squeeze::Squeeze(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Squeeze::validate() {
    const auto inputRank = getInputOperandDimensions(0).size();
    if (inputRank > 4) return false;

    if ( !isValidInputTensor(0)) {
         ALOGE("%s Empty  or Invalid dimensions size for input", __func__);
         return false;
    }
    // TODO: Add Support for all_tensors_as_inputs
    const auto& dimsOperandIndex1 = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);

    if (!sModelInfo->isOperandLifeTimeConst(dimsOperandIndex1)) {
        ALOGE("%s Only Constant dimensions supported now", __func__);
        return false;
    }
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    if (inputsSize == 2) {
        const auto& dimsOperandIndex2 = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
        // TODO: Support OmittedInput.
        // The empty 2nd argument in Squeeze op causes dynamic output
        // To add support, the dims will have to be calculated statically
        if (!isValidInputTensor(1) || !sModelInfo->isOperandLifeTimeConst(dimsOperandIndex2) ||
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 1) ) {
            ALOGE("%s Invalid operand type or operand lifetime", __func__);
            return false;
        }
    }

    return true;
}

std::shared_ptr<ngraph::Node> Squeeze::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;

    input = getInputNode(0);

    std::shared_ptr<ngraph::Node> dims;

    if (!sModelInfo->isOmittedInput(mNnapiOperationIndex, 1))
        dims = getInputNode(1);
    else
        dims = createConstNode(ngraph::element::i32, {0}, std::vector<int64_t>{});

    std::shared_ptr<ngraph::Node> outputNode;

    outputNode = std::make_shared<ngraph::opset3::Squeeze>(input, dims);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
