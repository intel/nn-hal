#include <OperationsFactory.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

OperationsFactory::OperationsFactory(const std::string& plugin, std::shared_ptr<NgraphNodes> nodes)
    : mNgraphNodes(nodes) {
    OperationsBase::sPluginType = plugin;
}
std::shared_ptr<OperationsBase> OperationsFactory::getOperation(const OperationType& type,
                                                                const Model& model) {
    auto opIter = mOperationsMap.find(type);
    if (opIter != mOperationsMap.end()) return opIter->second;

    switch (type) {
        case OperationType::ADD:
            mOperationsMap[type] = std::make_shared<Add>(model);
            break;
        case OperationType::CONCATENATION:
            mOperationsMap[type] = std::make_shared<Concat>(model);
            break;
        default:
            return nullptr;
    }
    mOperationsMap[type]->setNgraphNodes(mNgraphNodes);
    return mOperationsMap[type];
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android