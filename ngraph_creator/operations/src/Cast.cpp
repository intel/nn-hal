#include <Cast.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Cast::Cast(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Cast::validate() {
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

    return true;
}

std::shared_ptr<ngraph::Node> Cast::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        input = getInputNode<float>(0);
    }

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_INT32)) {
        input = getInputNode<int>(0);
    }

    ngraph::element::Type elementType;  // change to outputbased element type

    if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        elementType = ngraph::element::f32;
    }

    if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_INT32)) {
        elementType = ngraph::element::i32;
    }

    auto outputNode = std::make_shared<ngraph::opset3::Convert>(input, elementType);

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
