#include <NgraphNodes.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
NgraphNodes::NgraphNodes(size_t size) { mOperationOutputs.reserve(size); }

void NgraphNodes::addInputParam(size_t index, std::shared_ptr<ngraph::opset3::Parameter> inParam) {
    mInputParamsMap[index] = inParam;
}
void NgraphNodes::setOperationOutput(size_t index, ngraph::Output<ngraph::Node> output) {
    mOperationOutputs[index] = output;
}
ngraph::Output<ngraph::Node> NgraphNodes::getOperationOutput(size_t index) {
    return mOperationOutputs[index];
}

void NgraphNodes::setResultNode(size_t outputIndex) {
    // TODO: Construct this similar to other operations
    ngraph::AxisVector order = {0, 2, 3, 1};  // NCHW_NHWC
    const auto order_node =
        ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{order.size()}, order);
    std::shared_ptr<ngraph::Node> resultNode =
        std::make_shared<ngraph::opset3::Transpose>(mOperationOutputs[outputIndex], order_node);
    mResultsMap[outputIndex] = resultNode;
}

const std::string& NgraphNodes::getNodeName(size_t index) {
    // The getNodeName is expected to be called only for Inputs and Outputs.
    // Hence, scan through mInputParamsMap and mResultsMap to identify a valid node name.
    if (mResultsMap.find(index) != mResultsMap.end()) return mResultsMap[index]->get_name();
    if (mInputParamsMap.find(index) != mInputParamsMap.end())
        return mInputParamsMap[index]->get_name();
    return INVALID_STRING;
}

std::shared_ptr<ngraph::Function> NgraphNodes::generateGraph() {
    std::vector<std::shared_ptr<ngraph::Node>> resultNodes;
    resultNodes.reserve(mResultsMap.size());
    for (auto const& temp : mResultsMap) resultNodes.push_back(temp.second);
    // TODO: Remove the Dummy Concat
    // Dummy Concat to join the disconnected graph(ssd_mobilenet Obj Det with only Concat)
    resultNodes.push_back(std::make_shared<ngraph::opset3::Concat>(resultNodes, 3));
    ngraph::ParameterVector inputParams;
    inputParams.reserve(mInputParamsMap.size());
    for (auto const& temp : mInputParamsMap) inputParams.push_back(temp.second);
    return std::make_shared<ngraph::Function>(resultNodes, inputParams);
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
