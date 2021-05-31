#include <Mean.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Mean::Mean(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Mean::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) return false;

    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_INT32)) return false;

    if (!checkInputOperandType(2, (int32_t)OperandType::INT32)) return false;

    // TODO: Add Support for all_tensors_as_inputs
    const auto& axesOperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);

    if (!sModelInfo->isOperandLifeTimeConst(axesOperandIndex)) {
        ALOGE("%s Only Constant dimensions supported now", __func__);
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Mean::createNode() {
    // Creating input nodes
    auto input = getInputNode<float>(0);
    auto reduction_axes = getInputNode<int>(1);
    auto reduce_dims = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 2);
    bool keep_dims = (reduce_dims > 0) ? true : false;

    auto outputNode =
        std::make_shared<ngraph::opset3::ReduceMean>(input, reduction_axes, keep_dims);

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
