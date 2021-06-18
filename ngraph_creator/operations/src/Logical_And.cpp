#include <Logical_And.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Logical_And::Logical_And(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Logical_And::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_BOOL8)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_BOOL8)) {
        return false;
    }
    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_BOOL8)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Logical_And::createNode() {
    // Creating input nodes
    auto input1 = getInputNode(0);
    auto input2 = getInputNode(1);

    auto outputNode = std::make_shared<ngraph::opset3::LogicalAnd>(
        input1, input2, ngraph::op::AutoBroadcastType::NUMPY);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
