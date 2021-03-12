#pragma once

#include <Driver.h>
#include <Temp.h>  //TODO: Remove this once NNAPI_Utils is ready
#include <android/log.h>
#include <log/log.h>
#include <NgraphNodes.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset3.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class OperationsBase {
protected:
    enum ConversionType { NHWC_NCHW, NCHW_NHWC };
    uint32_t mDefaultOutputIndex;
    const Operation& mNnapiOp;
    std::shared_ptr<ngraph::Node> transpose(ConversionType type,
                                            ngraph::Output<ngraph::Node> input);
    virtual std::shared_ptr<ngraph::Node> createNode() = 0;
    // override createNodeForPlugin in case sPluginType specific implementation is required
    virtual std::shared_ptr<ngraph::Node> createNodeForPlugin();
    std::shared_ptr<ngraph::Node> toNCHW(size_t inputIndex, size_t outputIndex);
    void addResultNode(size_t index, std::shared_ptr<ngraph::Node> resultNode);

public:
    static Model* sModel;
    static std::shared_ptr<NgraphNodes> mNgraphNodes;
    static std::string sPluginType;
    OperationsBase(const Operation& op);
    void setNgraphNodes(std::shared_ptr<NgraphNodes> nodes);
    virtual bool validate();
    // override connectOperationToGraph in case Operation has multiple outputs
    virtual void connectOperationToGraph();
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android