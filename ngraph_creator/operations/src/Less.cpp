#include <Less.hpp>
#define LOG_TAG "Less"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Less::Less(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Less::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_BOOL8)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_INT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_BOOL8) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }
    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(1, (int32_t)OperandType::TENSOR_INT32) &&
        !!checkInputOperandType(1, (int32_t)OperandType::TENSOR_BOOL8) &&
        !checkInputOperandType(1, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Less::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input1, input2;

    input1 = getInputNode(0);
    input2 = getInputNode(1);

    std::shared_ptr<ngraph::Node> outputNode;

    outputNode = std::make_shared<ngraph::opset3::Less>(input1, input2,
                                                        ngraph::op::AutoBroadcastType::NUMPY);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
