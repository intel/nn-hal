#pragma once

#include <Temp.h>  //TODO: Remove this once NNAPI_Utils is ready
#include <cstdio>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset3.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class NgraphNodes {
private:
    std::vector<ngraph::Output<ngraph::Node>> mOperationOutputs;
    std::vector<bool> mForcedNchw;
    std::map<int, std::shared_ptr<ngraph::opset3::Parameter>> mInputParamsMap;
    std::vector<std::shared_ptr<ngraph::Node>> mResultNodes;

public:
    NgraphNodes(size_t operandsSize, size_t resultsSize);
    ~NgraphNodes();

    void addInputParam(size_t index, std::shared_ptr<ngraph::opset3::Parameter> inParam);
    void setOperationOutput(size_t index, ngraph::Output<ngraph::Node> output);
    ngraph::Output<ngraph::Node> getOperationOutput(size_t index);
    bool isForcedNchw(size_t index);
    void setForcedNchw(size_t index, bool flag);
    void setResultNode(size_t outputIndex, std::shared_ptr<ngraph::Node> resultNode);

    const std::string& getNodeName(size_t index);

    std::shared_ptr<ngraph::Function> generateGraph();
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
