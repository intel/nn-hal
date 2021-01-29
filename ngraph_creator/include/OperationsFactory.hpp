#include <Add.hpp>
#include <Concat.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class OperationsFactory {
private:
    std::map<OperationType, std::shared_ptr<OperationsBase>> mOperationsMap;
    std::shared_ptr<NgraphNodes> mNgraphNodes;

public:
    OperationsFactory(const std::string& plugin, std::shared_ptr<NgraphNodes> nodes);
    std::shared_ptr<OperationsBase> getOperation(const OperationType& type, const Model& model);
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android