#include <Add.hpp>
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

    const auto& activationIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 2);
    if (!sModelInfo->isOperandLifeTimeConst(activationIndex)) {
        ALOGE("%s Due to OpenVINO API restrictions, Scalar input values must have CONST lifetime",
              __func__);
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Add::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input1, input2;

    input1 = getInputNode(0);
    input2 = getInputNode(1);

    auto activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);

    auto addNode =
        std::make_shared<ngraph::opset3::Add>(input1, input2, ngraph::op::AutoBroadcastType::NUMPY);

    auto outputNode = applyActivation(addNode, activationFn);

    return outputNode;
}

std::shared_ptr<ngraph::Node> Add::createNodeForPlugin() {
#if 0
    if (sPluginType == IntelDeviceType::VPU) {
        auto input = mNgraphNodes->getOperationOutput(
            sModelInfo->getOperationInput(mNnapiOperationIndex, 0));
        std::shared_ptr<ngraph::Node> constantOp =
            std::make_shared<ngraph::opset3::Constant>(ngraph::element::f32, input.get_shape());
        auto transposedOp = transpose(NHWC_NCHW, constantOp);
        return std::make_shared<ngraph::opset3::Add>(input, transposedOp,
                                                     ngraph::op::AutoBroadcastType::NUMPY);
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
