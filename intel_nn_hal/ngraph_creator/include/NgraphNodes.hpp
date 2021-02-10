#pragma once

#include <cstdio>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset3.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class NgraphNodes {
private:
    std::vector<std::shared_ptr<ngraph::Node>> mOperationOutputs;
    ngraph::ParameterVector mInputParams;
    std::vector<std::shared_ptr<ngraph::Node>> mResultNodes;

public:
    NgraphNodes(size_t size);

    void addInputParam(std::shared_ptr<ngraph::opset3::Parameter> inParam);
    void setOperationOutput(size_t index, std::shared_ptr<ngraph::Node> node);
    std::shared_ptr<ngraph::Node> getOperationOutput(size_t index);
    void setResultNode(size_t index);

    const std::string& getNodeName(size_t index);

    std::shared_ptr<ngraph::Function> generateGraph();
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
