#include <Add.hpp>
#include "openvino/opsets/opset8.hpp"
#undef LOG_TAG
#define LOG_TAG "Add"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Add::Add(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Add::validate() {
    ALOGV("%s PASSED", __func__);

    return true;
}

std::shared_ptr<ov::Node> Add::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input1, input2;
    std::shared_ptr<ov::Node> outputNode;

    input1 = getInputNode(0);
    input2 = getInputNode(1);

    const auto& activationIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 2);
    if (sModelInfo->isOperandLifeTimeConst(activationIndex)) {
        auto activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);
        auto addNode =
            std::make_shared<ov::opset8::Add>(input1, input2, ov::op::AutoBroadcastType::NUMPY);
        outputNode = applyActivation(addNode, activationFn);
    } else {
        auto addNode =
            std::make_shared<ov::opset3::Add>(input1, input2, ov::op::AutoBroadcastType::NUMPY);
        outputNode = addNode;
    }

    return outputNode;
}

std::shared_ptr<ov::Node> Add::createNodeForPlugin() {
#if 0
    if (sPluginType == IntelDeviceType::VPU) {
        auto input = mNgraphNodes->getOperationOutput(
            sModelInfo->getOperationInput(mNnapiOperationIndex, 0));
        std::shared_ptr<ov::Node> constantOp =
            std::make_shared<ov::opset3::Constant>(ov::element::f32, input.get_shape());
        auto transposedOp = transpose(NHWC_NCHW, constantOp);
        return std::make_shared<ov::opset3::Add>(input, transposedOp,
                                                     ov::op::AutoBroadcastType::NUMPY);
    } else {
        return createNode();
    }
#endif
    return createNode();
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
