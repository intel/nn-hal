#include <Add.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Add::Add(const Model& model) : OperationsBase(model) {}

bool Add::validate(const Operation& op) { return true; }

// TODO: Implement APIs createNode & createNodeForPlugin. These are just dummy placeholders.
std::shared_ptr<ngraph::Node> Add::createNode(const Operation& operation) {
    auto input = mNgraphNodes->getOperationOutput(operation.inputs[OP_INPUT_IDX_CONV]);
    std::shared_ptr<ngraph::Node> constantOp =
        std::make_shared<ngraph::opset3::Constant>(ngraph::element::f32, input.get_shape());
    auto transposedOp = transpose(NHWC_NCHW, constantOp);
    return std::make_shared<ngraph::opset3::Add>(input, transposedOp,
                                                 ngraph::op::AutoBroadcastType::NUMPY);
}

std::shared_ptr<ngraph::Node> Add::createNodeForPlugin(const Operation& operation) {
    if (sPluginType == "VPU") {
        auto input = mNgraphNodes->getOperationOutput(operation.inputs[OP_INPUT_IDX_CONV]);
        std::shared_ptr<ngraph::Node> constantOp =
            std::make_shared<ngraph::opset3::Constant>(ngraph::element::f32, input.get_shape());
        auto transposedOp = transpose(NHWC_NCHW, constantOp);
        return std::make_shared<ngraph::opset3::Add>(input, transposedOp,
                                                     ngraph::op::AutoBroadcastType::NUMPY);
    } else {
        return createNode(operation);
    }
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android