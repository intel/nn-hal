#include <Add.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Add::Add(int operationIndex) : OperationsBase(operationIndex) {}

bool Add::validate() { return true; }

// TODO: Implement APIs createNode & createNodeForPlugin. These are just dummy placeholders.
std::shared_ptr<ngraph::Node> Add::createNode() {
    auto input = mNgraphNodes->getOperationOutput(
        sModelInfo->getOperationInput(mNnapiOperationIndex, OP_INPUT_IDX_CONV));
    std::shared_ptr<ngraph::Node> constantOp =
        std::make_shared<ngraph::opset3::Constant>(ngraph::element::f32, input.get_shape());
    auto transposedOp = transpose(NHWC_NCHW, constantOp);
    return std::make_shared<ngraph::opset3::Add>(input, transposedOp,
                                                 ngraph::op::AutoBroadcastType::NUMPY);
}

std::shared_ptr<ngraph::Node> Add::createNodeForPlugin() {
    if (sPluginType == "VPU") {
        auto input = mNgraphNodes->getOperationOutput(
            sModelInfo->getOperationInput(mNnapiOperationIndex, OP_INPUT_IDX_CONV));
        std::shared_ptr<ngraph::Node> constantOp =
            std::make_shared<ngraph::opset3::Constant>(ngraph::element::f32, input.get_shape());
        auto transposedOp = transpose(NHWC_NCHW, constantOp);
        return std::make_shared<ngraph::opset3::Add>(input, transposedOp,
                                                     ngraph::op::AutoBroadcastType::NUMPY);
    } else {
        return createNode();
    }
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
