#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

std::string OperationsBase::sPluginType;

std::shared_ptr<ngraph::Node> OperationsBase::transpose(ConversionType type,
                                                        ngraph::Output<ngraph::Node> input) {
    ngraph::AxisVector order;
    switch (type) {
        case NHWC_NCHW:
            order = {0, 3, 1, 2};
            break;
        case NCHW_NHWC:
            order = {0, 2, 3, 1};
    }
    const auto order_node =
        ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{order.size()}, order);
    return std::make_shared<ngraph::opset3::Transpose>(input, order_node);
}

// override createNodeForPlugin in case sPluginType specific implementation is required
std::shared_ptr<ngraph::Node> OperationsBase::createNodeForPlugin(const Operation& op) {
    return createNode(op);
}

// override connectOperationToGraph in case Operation has multiple outputs
void OperationsBase::connectOperationToGraph(const Operation& op) {
    mNgraphNodes->setOperationOutput(op.outputs[0], createNodeForPlugin(op)->get_default_output());
}

OperationsBase::OperationsBase(const Model& model) : mModel(model) {}

void OperationsBase::setNgraphNodes(std::shared_ptr<NgraphNodes> nodes) { mNgraphNodes = nodes; }

bool OperationsBase::validate(const Operation& op) { return true; }

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android