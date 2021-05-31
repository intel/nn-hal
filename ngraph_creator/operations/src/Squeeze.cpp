#include <Squeeze.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Squeeze::Squeeze(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Squeeze::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) return false;

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) return false;

    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_INT32)) return false;

    // TODO: Add Support for all_tensors_as_inputs
    const auto& dimsOperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);

    // TODO: Support OmittedInput.
    // The empty 2nd argument in Squeeze op causes dynamic output
    // To add support, the dims will have to be calculated statically
    if (sModelInfo->isOmittedInput(mNnapiOperationIndex, 1) ||
        !sModelInfo->isOperandLifeTimeConst(dimsOperandIndex)) {
        ALOGE("%s Only Constant dimensions supported now", __func__);
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Squeeze::createNode() {
    // Creating input nodes
    auto input = getInputNode<float>(0);

    std::shared_ptr<ngraph::Node> dims;

    if (!sModelInfo->isOmittedInput(mNnapiOperationIndex, 1))
        dims = getInputNode<int>(1);
    else
        dims = make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{0},
                                                     std::vector<int64_t>{});

    auto outputNode = std::make_shared<ngraph::opset3::Squeeze>(input, dims);

    const auto op = sModelInfo->getOperand(mDefaultOutputIndex);
    if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
        addResultNode(mDefaultOutputIndex, outputNode);
    }
    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
