//#define LOG_NDEBUG 0
#include <NgraphNodes.hpp>
#define LOG_TAG "NgraphNodes"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
NgraphNodes::NgraphNodes(size_t operandsSize, size_t resultsSize) {
    mOperationOutputs.resize(operandsSize);
    mForcedNchw.assign(operandsSize, false);
    mResultNodes.reserve(resultsSize);
    ALOGV("%s Constructed operandsSize %d, resultsSize %d", __func__, operandsSize, resultsSize);
}

NgraphNodes::~NgraphNodes() { ALOGV("%s Destructed", __func__); }

void NgraphNodes::addInputParam(size_t index, std::shared_ptr<ngraph::opset3::Parameter> inParam) {
    mInputParamsMap[index] = inParam;
}
void NgraphNodes::setOperationOutput(size_t index, ngraph::Output<ngraph::Node> output) {
    mOperationOutputs[index] = output;
}
ngraph::Output<ngraph::Node> NgraphNodes::getOperationOutput(size_t index) {
    return mOperationOutputs[index];
}
bool NgraphNodes::isForcedNchw(size_t index) {
    return mForcedNchw[index];
}
void NgraphNodes::setForcedNchw(size_t index, bool flag) {
    mForcedNchw[index] = flag;
}

void NgraphNodes::setResultNode(size_t outputIndex, std::shared_ptr<ngraph::Node> resultNode) {
    ALOGD("setResultNode %d", outputIndex);
    mResultNodes.push_back(resultNode);
}

const std::string& NgraphNodes::getNodeName(size_t index) {
    if(mNodeNames.find(index) == mNodeNames.end())
        mNodeNames[index] = mOperationOutputs[index].get_node_shared_ptr()->get_name();
    ALOGV("%s index %d, name %s", __func__, index, mNodeNames[index].c_str());
    return mNodeNames[index];
}

std::shared_ptr<ngraph::Function> NgraphNodes::generateGraph() {
    // TODO: Remove the Dummy Concat
    // Dummy Concat to join the disconnected graph(ssd_mobilenet Obj Det with only Concat)
    mResultNodes.push_back(std::make_shared<ngraph::opset3::Concat>(mResultNodes, 2));
    ngraph::ParameterVector inputParams;
    inputParams.reserve(mInputParamsMap.size());
    for (auto const& temp : mInputParamsMap) inputParams.push_back(temp.second);
    return std::make_shared<ngraph::Function>(mResultNodes, inputParams);
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
