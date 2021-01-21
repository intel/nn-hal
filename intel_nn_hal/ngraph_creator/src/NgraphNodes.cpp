#include <NgraphNodes.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
NgraphNodes::NgraphNodes(size_t size) : INVALID_STRING("Invalid Node") {
    mOperationOutputs.reserve(size);
}

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
    // Using Convert instead of Result due to crash with libinference_engine_legacy
    std::shared_ptr<ngraph::Node> resultNode = std::make_shared<ngraph::opset3::Convert>(
        mOperationOutputs[outputIndex], ngraph::element::f32);
    mResultsMap[outputIndex] = resultNode;
}

const std::string& NgraphNodes::getNodeName(size_t index) {
    if (mResultsMap.find(index) != mResultsMap.end()) return mResultsMap[index]->get_name();
    if (mInputParamsMap.find(index) != mInputParamsMap.end())
        return mInputParamsMap[index]->get_name();
    return INVALID_STRING;
}

std::shared_ptr<ngraph::Function> NgraphNodes::generateGraph() {
    std::vector<std::shared_ptr<ngraph::Node>> resultNodes;
    resultNodes.reserve(mResultsMap.size() + 1);
    for (auto const& temp : mResultsMap) resultNodes.push_back(temp.second);
    resultNodes.push_back(std::make_shared<ngraph::opset3::Concat>(
        resultNodes, 1));  // Dummy Concat to join the network
    ngraph::ParameterVector inputParams;
    inputParams.reserve(mInputParamsMap.size());
    for (auto const& temp : mInputParamsMap) inputParams.push_back(temp.second);
    return std::make_shared<ngraph::Function>(resultNodes, inputParams);
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
