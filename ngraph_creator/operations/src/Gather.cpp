#include <Gather.hpp>
#define LOG_TAG "Gather"

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
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_INT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_INT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
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

    gatherVals = getInputNode(0);

    // axis range [-n, n]
    auto axis = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);
    auto axisNode = createConstNode(ngraph::element::i32, {}, convertToVector(axis));

    auto indices = getInputNode(2);

    std::shared_ptr<ngraph::Node> outputNode;
    outputNode = std::make_shared<ngraph::opset3::Gather>(gatherVals, indices, axisNode);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
