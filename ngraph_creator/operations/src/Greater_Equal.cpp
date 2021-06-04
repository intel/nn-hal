#include <Greater_Equal.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Greater_Equal::Greater_Equal(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Greater_Equal::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_BOOL8)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_INT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_BOOL8)) {
        return false;
    }
    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(1, (int32_t)OperandType::TENSOR_INT32) &&
        !checkInputOperandType(1, (int32_t)OperandType::TENSOR_BOOL8)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Greater_Equal::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input1, input2;

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        input1 = getInputNode<float>(0);
        input2 = getInputNode<float>(1);
    }

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_INT32)) {
        input1 = getInputNode<int>(0);
        input2 = getInputNode<int>(1);
    }

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_BOOL8)) {
        input1 = getInputNode<bool>(0);
        input2 = getInputNode<bool>(1);
    }

    auto outputNode = std::make_shared<ngraph::opset3::GreaterEqual>(
        input1, input2, ngraph::op::AutoBroadcastType::NUMPY);

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
