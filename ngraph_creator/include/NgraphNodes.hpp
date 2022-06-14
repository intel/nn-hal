#pragma once

#include <log/log.h>
#include <cstdio>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset3.hpp>

#undef LOG_TAG

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class NgraphNodes {
private:
    std::vector<ngraph::Output<ngraph::Node>> mOutputAtOperandIndex;
    // mForcedNchw flag tracks whether a forced conversion to NCHW has been done at ngraph_creator
    // in the path to current Operand.
    std::vector<bool> mForcedNchw;
    std::vector<std::shared_ptr<ngraph::opset3::Parameter>> mInputParams;
    std::vector<std::shared_ptr<ngraph::Node>> mResultNodes;
    // mNodeNames are only populated when requested, as only Inputs and Result NodeNames are
    // required.
    std::map<int, std::string> mNodeNames;

public:
    NgraphNodes(size_t operandsSize, size_t resultsSize);
    ~NgraphNodes();

    void addInputParam(std::shared_ptr<ngraph::opset3::Parameter> inParam);
    void setOutputAtOperandIndex(size_t index, ngraph::Output<ngraph::Node> output);
    ngraph::Output<ngraph::Node> getOperationOutput(size_t index);
    void setResultNode(size_t outputIndex, std::shared_ptr<ngraph::Node> resultNode);

    const std::string& getNodeName(size_t index);
    void removeInputParameter(std::string name, size_t index);
    std::vector<size_t> getOutputShape(size_t index);

    std::shared_ptr<ngraph::Function> generateGraph();
    // Setting the node name to empty string "". Caller of getNodeName should validate against "".
    void setInvalidNode(size_t index);
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
