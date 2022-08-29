#include <Less.hpp>
#undef LOG_TAG
#define LOG_TAG "Less"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Less::Less(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Less::validate() {
    auto operandIndex1 = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
    auto operandIndex2 = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
    const auto& elementType1 = sModelInfo->getOperandType(operandIndex1);
    const auto& elementType2 = sModelInfo->getOperandType(operandIndex2);
    if ( !isValidInputTensor(0) || !isValidInputTensor(1) ) {
         ALOGE("%s Empty  or Invalid dimensions size for input", __func__);
         return false;
    }
    //check operand lifetime
    if(!sModelInfo->isOperandLifeTimeConst(operandIndex1) ||
        !sModelInfo->isOperandLifeTimeConst(operandIndex2)) {
        ALOGE("%s Only Const lifetime is supported", __func__);
        return false;
    }

    // check if both tensors are of same type
    if(elementType1 != elementType2 ) {
        ALOGE("%s Input type mismatch", __func__);
        return false;
    }

    ALOGV("%s PASSED", __func__);
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
