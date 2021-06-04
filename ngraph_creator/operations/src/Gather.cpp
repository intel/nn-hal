#include <Gather.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Gather::Gather(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Gather::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_INT32)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_INT32)) {
        return false;
    }

    if (!checkInputOperandType(1, (int32_t)OperandType::INT32)) {
        return false;
    }

    if (!checkInputOperandType(2, (int32_t)OperandType::TENSOR_INT32)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Gather::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> gatherVals;

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        gatherVals = getInputNode<float>(0);
    }

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_INT32)) {
        gatherVals = getInputNode<int>(0);
    }
    // axis range [-n, n]
    auto axis = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);
    auto axisNode = ngraph::opset3::Constant::create(ngraph::element::i64, {}, {axis});
    auto indices = getInputNode<int>(2);

    auto outputNode = std::make_shared<ngraph::opset3::Gather>(gatherVals, indices, axisNode);

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
