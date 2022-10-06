#include <NgraphNodes.hpp>
#undef LOG_TAG
#define LOG_TAG "NgraphNodes"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
NgraphNodes::NgraphNodes(size_t operandsSize, size_t resultsSize) {
    mOutputAtOperandIndex.resize(operandsSize);
    mForcedNchw.assign(operandsSize, false);
    mResultNodes.reserve(resultsSize);
    ALOGV("%s Constructed operandsSize %zu, resultsSize %zu", __func__, operandsSize, resultsSize);
}

NgraphNodes::~NgraphNodes() { ALOGV("%s Destructed", __func__); }

void NgraphNodes::addInputParam(std::shared_ptr<ov::opset3::Parameter> inParam) {
    mInputParams.push_back(inParam);
}
void NgraphNodes::setOutputAtOperandIndex(size_t index, ov::Output<ov::Node> output) {
    ALOGV("%s index %zu", __func__, index);
    mOutputAtOperandIndex[index] = output;
}
ov::Output<ov::Node> NgraphNodes::getOperationOutput(size_t index) {
    return mOutputAtOperandIndex[index];
}

void NgraphNodes::setResultNode(size_t outputIndex, std::shared_ptr<ov::Node> resultNode) {
    ALOGD("setResultNode %zu", outputIndex);
    mResultNodes.push_back(resultNode);
}

const std::string& NgraphNodes::getNodeName(size_t index) {
    if (mNodeNames.find(index) == mNodeNames.end()) {
        mNodeNames[index] = mOutputAtOperandIndex[index].get_node_shared_ptr()->get_name();
        ALOGD("%s index %zu, name %s", __func__, index, mNodeNames[index].c_str());
    }
    ALOGV("%s index %zu, name %s", __func__, index, mNodeNames[index].c_str());
    return mNodeNames[index];
}
// remove null input node parameter
void NgraphNodes::removeInputParameter(std::string name, size_t index) {
    for (size_t i = 0; i < mInputParams.size(); i++) {
        if (name.compare(mInputParams[i]->get_name()) == 0) {
            mInputParams.erase(mInputParams.begin() + i);
            setInvalidNode(index);
        }
    }
}

std::shared_ptr<ov::Model> NgraphNodes::generateGraph() {
    return std::make_shared<ov::Model>(mResultNodes, mInputParams);
}

void NgraphNodes::setInvalidNode(size_t index) { mNodeNames[index] = ""; }

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
