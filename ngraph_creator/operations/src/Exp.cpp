#include <Exp.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Exp::Exp(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Exp::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Exp::createNode() {
    // Creating input nodes
    auto input = getInputNode<float>(0);

    auto outputNode = std::make_shared<ngraph::opset3::Exp>(input);

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
