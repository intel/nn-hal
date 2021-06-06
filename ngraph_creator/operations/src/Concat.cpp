//#define LOG_NDEBUG 0
#include <Concat.hpp>
#define LOG_TAG "Concat"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Concat::Concat(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Concat::validate() {
    // Check Output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }
    // check concatenation axis
    auto n = sModelInfo->getOperationInputsSize(mNnapiOperationIndex) -
             1;  // 0 ~ n-1: The list of n input tensors
    if (!checkInputOperandType(n, (int32_t)OperandType::INT32)) {
        return false;
    }
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Concat::createNode() {
    auto n = sModelInfo->getOperationInputsSize(mNnapiOperationIndex) -
             1;  // 0 ~ n-1: The list of n input tensors
    auto axis = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex,
                                                          n);  // n: concatenation axis
    std::vector<ngraph::Output<ngraph::Node>> inputs;
    ALOGD("createNode n %d, axis %d", n, axis);
    for (int i = 0; i < n; i++) {
        auto inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, i);
        auto inputOp = getInputNode(inputIndex);
        const auto op = sModelInfo->getOperand(inputIndex);
        ALOGD("createNode inputIndex %d, lifetime %d", inputIndex, op.lifetime);
        if (mNgraphNodes->isForcedNchw(inputIndex)) {
            inputOp = transpose(NCHW_NHWC, inputOp);
            mNgraphNodes->setForcedNchw(mDefaultOutputIndex, false);
        }
        inputs.push_back(inputOp);
    }

    std::shared_ptr<ngraph::Node> outputNode =
        std::make_shared<ngraph::opset3::Concat>(inputs, axis);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
