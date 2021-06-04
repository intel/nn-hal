#include <Logical_Not.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Logical_Not::Logical_Not(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Logical_Not::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_BOOL8)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_BOOL8)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Logical_Not::createNode() {
    // Creating input nodes
    auto input = getInputNode<bool>(0);

    auto outputNode = std::make_shared<ngraph::opset3::LogicalNot>(input);

    const auto op = sModelInfo->getOperand(mDefaultOutputIndex);
    if (op.lifetime == V1_3::OperandLifeTime::SUBGRAPH_OUTPUT) {
        addResultNode(mDefaultOutputIndex, outputNode);
    }
    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
