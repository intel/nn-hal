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
    std::map<int, std::shared_ptr<ngraph::opset3::Parameter>> mInputParamsMap;
    std::map<int, std::shared_ptr<ngraph::Node>> mResultsMap;

public:
    NgraphNodes(size_t size);

    void addInputParam(size_t index, std::shared_ptr<ngraph::opset3::Parameter> inParam);
    void setOperationOutput(size_t index, ngraph::Output<ngraph::Node> output);
    ngraph::Output<ngraph::Node> getOperationOutput(size_t index);
    void setResultNode(size_t outputIndex);

    const std::string& getNodeName(size_t index);

    std::shared_ptr<ngraph::Function> generateGraph();
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
