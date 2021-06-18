#include <Sub.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Sub::Sub(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Sub::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    // Check all input types
    for (int i = 0; i <= 1; i++) {
        if (!checkInputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32) &&
            !checkInputOperandType(i, (int32_t)OperandType::TENSOR_QUANT8_ASYMM))
            return false;
        ;
    }

    if (!checkInputOperandType(2, (int32_t)OperandType::INT32)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Sub::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input1, input2;

    input1 = getInputNode(0);
    input2 = getInputNode(1);

    auto activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);

    auto subNode = std::make_shared<ngraph::opset3::Subtract>(input1, input2,
                                                              ngraph::op::AutoBroadcastType::NUMPY);

    auto outputNode = applyActivation(subNode, activationFn);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
