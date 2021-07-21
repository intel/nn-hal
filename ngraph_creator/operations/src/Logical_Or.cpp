#include <Logical_Or.hpp>
#define LOG_TAG "Logical_Or"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Logical_Or::Logical_Or(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Logical_Or::validate() {
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

std::shared_ptr<ngraph::Node> Logical_Or::createNode() {
    // Creating input nodes
    auto input1 = getInputNode(0);
    auto input2 = getInputNode(1);

    auto outputNode = std::make_shared<ngraph::opset3::LogicalOr>(
        input1, input2, ngraph::op::AutoBroadcastType::NUMPY);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
