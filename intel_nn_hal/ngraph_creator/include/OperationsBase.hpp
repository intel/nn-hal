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
    Model mModel;
    std::shared_ptr<NgraphNodes> mNgraphNodes;
    enum ConversionType { NHWC_NCHW, NCHW_NHWC };
    std::shared_ptr<ngraph::Node> transpose(ConversionType type,
                                            std::shared_ptr<ngraph::Node> input);
    virtual std::shared_ptr<ngraph::Node> createNode(const Operation& op) = 0;

public:
    static std::string sPluginType;
    OperationsBase(const Model& model);
    void setNgraphNodes(std::shared_ptr<NgraphNodes> nodes);
    virtual bool validate(const Operation& op);

    // override createNodeForPlugin in case sPluginType specific implementation is required
    virtual std::shared_ptr<ngraph::Node> createNodeForPlugin(const Operation& op);
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android