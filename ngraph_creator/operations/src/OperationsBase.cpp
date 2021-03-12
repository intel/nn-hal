#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

std::string OperationsBase::sPluginType;
std::shared_ptr<NgraphNodes> OperationsBase::mNgraphNodes;
Model* OperationsBase::sModel;

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

std::shared_ptr<ngraph::Node> OperationsBase::toNCHW(size_t inputIndex, size_t outputIndex) {
    auto inNode = mNgraphNodes->getOperationOutput(inputIndex).get_node_shared_ptr();
    if (mNgraphNodes->isForcedNchw(inputIndex))
        return inNode;
    else {
        mNgraphNodes->setForcedNchw(outputIndex, true);
        return transpose(NHWC_NCHW, inNode);
    }
}

// override createNodeForPlugin in case sPluginType specific implementation is required
std::shared_ptr<ngraph::Node> OperationsBase::createNodeForPlugin() { return createNode(); }

// override connectOperationToGraph in case Operation has multiple outputs
void OperationsBase::connectOperationToGraph() {
    mNgraphNodes->setOutputAtOperandIndex(mDefaultOutputIndex,
                                          createNodeForPlugin()->get_default_output());
}

void OperationsBase::addResultNode(size_t index, std::shared_ptr<ngraph::Node> resultNode) {
    mNgraphNodes->setResultNode(index, resultNode);
}

OperationsBase::OperationsBase(const Operation& op) : mNnapiOp(op) { mDefaultOutputIndex = 0; }

void OperationsBase::setNgraphNodes(std::shared_ptr<NgraphNodes> nodes) { mNgraphNodes = nodes; }

bool OperationsBase::validate() { return true; }

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android