//#define LOG_NDEBUG 0
#include <NgraphNodes.hpp>
#define LOG_TAG "NgraphNodes"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
NgraphNodes::NgraphNodes(size_t operandsSize, size_t resultsSize) {
    mOutputAtOperandIndex.resize(operandsSize);
    mForcedNchw.assign(operandsSize, false);
    mResultNodes.reserve(resultsSize);
    ALOGV("%s Constructed operandsSize %d, resultsSize %d", __func__, operandsSize, resultsSize);
}

NgraphNodes::~NgraphNodes() { ALOGV("%s Destructed", __func__); }

void NgraphNodes::addInputParam(std::shared_ptr<ngraph::opset3::Parameter> inParam) {
    mInputParams.push_back(inParam);
}
void NgraphNodes::setOutputAtOperandIndex(size_t index, ngraph::Output<ngraph::Node> output) {
    ALOGV("%s index %d", __func__, index);
    mOutputAtOperandIndex[index] = output;
}
ngraph::Output<ngraph::Node> NgraphNodes::getOperationOutput(size_t index) {
    return mOutputAtOperandIndex[index];
}
bool NgraphNodes::isForcedNchw(size_t index) { return mForcedNchw[index]; }
void NgraphNodes::setForcedNchw(size_t index, bool flag) { mForcedNchw[index] = flag; }

void NgraphNodes::setResultNode(size_t outputIndex, std::shared_ptr<ngraph::Node> resultNode) {
    ALOGD("setResultNode %uz", outputIndex);
    mResultNodes.push_back(resultNode);
}

const std::string& NgraphNodes::getNodeName(size_t index) {
    if (mNodeNames.find(index) == mNodeNames.end()) {
        mNodeNames[index] = mOutputAtOperandIndex[index].get_node_shared_ptr()->get_name();
        ALOGD("%s index %d, name %s", __func__, index, mNodeNames[index].c_str());
    }
    ALOGV("%s index %d, name %s", __func__, index, mNodeNames[index].c_str());
    return mNodeNames[index];
}
// remove null input node parameter
const void NgraphNodes::removeInputParameter(std::string name, size_t index) {
    for (int i = 0; i < mInputParams.size(); i++) {
        if (name.compare(mInputParams[i]->get_name()) == 0) {
            mInputParams.erase(mInputParams.begin() + i);
            setInvalidNode(index);
        }
    }
}

std::shared_ptr<ngraph::Function> NgraphNodes::generateGraph() {
    return std::make_shared<ngraph::Function>(mResultNodes, mInputParams);
}

void NgraphNodes::setInvalidNode(size_t index) { mNodeNames[index] = ""; }

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
