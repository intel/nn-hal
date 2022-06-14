#pragma once

#include <NgraphNodes.hpp>
#include <OperationsFactory.hpp>
#include <ngraph/node.hpp>
#include "ModelManager.h"
#include "OperationsBase.hpp"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class NgraphNetworkCreator {
private:
    std::shared_ptr<NnapiModelInfo> mModelInfo;
    std::vector<std::shared_ptr<OperationsBase>> mOperationNodes;
    std::shared_ptr<NgraphNodes> mNgraphNodes;
    OperationsFactory mOpFactoryInstance;
    bool createInputParams();
    bool initializeModel();

public:
    NgraphNetworkCreator(std::shared_ptr<NnapiModelInfo> modelInfo, IntelDeviceType deviceType);
    ~NgraphNetworkCreator();
    void getSupportedOperations(std::vector<bool>& supportedOperations);
    bool validateOperations();

    const std::string& getNodeName(uint32_t index);
    std::vector<size_t> getOutputShape(uint32_t index);

    std::shared_ptr<ngraph::Function> generateGraph();
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
