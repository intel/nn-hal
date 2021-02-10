#include <NgraphNodes.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
NgraphNodes::NgraphNodes(size_t size) { mOperationOutputs.reserve(size); }

void NgraphNodes::addInputParam(std::shared_ptr<ngraph::opset3::Parameter> inParam) {
    mInputParams.push_back(inParam);
}
void NgraphNodes::setOperationOutput(size_t index, std::shared_ptr<ngraph::Node> node) {
    mOperationOutputs[index] = node;
}
std::shared_ptr<ngraph::Node> NgraphNodes::getOperationOutput(size_t index) {
    return mOperationOutputs[index];
}

void NgraphNodes::setResultNode(size_t index) { mResultNodes.push_back(mOperationOutputs[index]); }

const std::string& NgraphNodes::getNodeName(size_t index) {
    return mOperationOutputs[index]->get_name();
}

std::shared_ptr<ngraph::Function> NgraphNodes::generateGraph() {
    mResultNodes.push_back(std::make_shared<ngraph::opset3::Concat>(
        mResultNodes, 1));  // Dummy Concat to join the network
    return std::make_shared<ngraph::Function>(mResultNodes, mInputParams);
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
