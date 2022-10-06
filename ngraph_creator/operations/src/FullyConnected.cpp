#include <FullyConnected.hpp>
#undef LOG_TAG
#define LOG_TAG "FullyConnected"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

FullyConnected::FullyConnected(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

// Supports only FP32 input. Will add support for QUANT8 through decompose node
// once the vpu and gna plugin support if confirmed
bool FullyConnected::validate() {
    auto input0 = getInputOperand(0);

    if (isZeroSizedInput(0)) {
        ALOGE("%s Batch size of 0 is not supported", __func__);
        return false;
    }

    if (input0.dimensions.size() < 2) {
        ALOGE("%s Invalid input parameter dimensions!!!", __func__);
        return false;
    }

    ALOGD("%s succeeded", __func__);
    return true;
}

std::shared_ptr<ov::Node> FullyConnected::createNode() {
    std::shared_ptr<ov::Node> inputNode = getInputNode(0);
    std::shared_ptr<ov::Node> weightsNode = getInputNode(1);
    std::shared_ptr<ov::Node> biasNode, multiplyNode, addNode, activationNode;

    auto inputDims = getInputOperand(0).dimensions;
    auto weightDims = getInputOperand(1).dimensions;
    auto biasDims = getInputOperand(2).dimensions;

    if ((inputDims.size() > 2) || (inputDims[1] != weightDims[1])) {
        std::vector<size_t> newShape = {getNumberOfElements(inputDims) / weightDims[1],
                                        weightDims[1]};
        auto reshapeConstant = createConstNode(ov::element::i32, {2}, newShape);
        auto reshapeNode = std::make_shared<ov::op::v1::Reshape>(inputNode, reshapeConstant, false);
        multiplyNode = std::make_shared<ov::opset3::MatMul>(reshapeNode, weightsNode, false, true);
    } else {
        multiplyNode = std::make_shared<ov::opset3::MatMul>(inputNode, weightsNode, false, true);
    }

    if (!sModelInfo->isOmittedInput(mNnapiOperationIndex, 2) && biasDims.size() != 0) {
        biasNode = getInputNode(2);

        if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM) ||
            checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM_SIGNED))
            biasNode = DequantizeNode(
                biasNode, sModelInfo->getOperationInput(mNnapiOperationIndex, 2), ov::element::f32);

        addNode = std::make_shared<ov::opset3::Add>(multiplyNode, biasNode,
                                                    ov::op::AutoBroadcastType::NUMPY);
    } else {
        ALOGD("FullyConnected: Bias not provided !!!");
        addNode = multiplyNode;
    }

    auto activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 3);
    activationNode = applyActivation(addNode, activationFn);
    return activationNode ? activationNode : addNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
