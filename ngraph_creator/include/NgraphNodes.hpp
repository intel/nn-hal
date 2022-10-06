#pragma once

#include <log/log.h>
#include <cstdio>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset3.hpp>

#undef LOG_TAG

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class NgraphNodes {
private:
    std::vector<ov::Output<ov::Node>> mOutputAtOperandIndex;
    // mForcedNchw flag tracks whether a forced conversion to NCHW has been done at ngraph_creator
    // in the path to current Operand.
    std::vector<bool> mForcedNchw;
    std::vector<std::shared_ptr<ov::opset3::Parameter>> mInputParams;
    std::vector<std::shared_ptr<ov::Node>> mResultNodes;
    // mNodeNames are only populated when requested, as only Inputs and Result NodeNames are
    // required.
    std::map<int, std::string> mNodeNames;

public:
    NgraphNodes(size_t operandsSize, size_t resultsSize);
    ~NgraphNodes();

    void addInputParam(std::shared_ptr<ov::opset3::Parameter> inParam);
    void setOutputAtOperandIndex(size_t index, ov::Output<ov::Node> output);
    ov::Output<ov::Node> getOperationOutput(size_t index);
    void setResultNode(size_t outputIndex, std::shared_ptr<ov::Node> resultNode);

    const std::string& getNodeName(size_t index);
    void removeInputParameter(std::string name, size_t index);

    std::shared_ptr<ov::Model> generateGraph();
    // Setting the node name to empty string "". Caller of getNodeName should validate against "".
    void setInvalidNode(size_t index);
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
