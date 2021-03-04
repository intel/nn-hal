#include <NgraphNodes.hpp>
#include <OperationsFactory.hpp>
//#include "ModelManger.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class NgraphNetworkCreator {
private:
    NnapiModelInfo* mModelInfo;
    std::shared_ptr<NgraphNodes> mNgraphNodes;
    OperationsFactory mOpFctryInst;
    void createInputParams();

public:
    NgraphNetworkCreator(NnapiModelInfo* model, const std::string& plugin);

    bool validateOperations();
    bool initializeModel();

    const std::string& getNodeName(uint32_t index);

    std::shared_ptr<ngraph::Function> generateGraph();
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
