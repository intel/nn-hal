#include <OperationsFactory.hpp>
#define LOG_TAG "OperationsFactory"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

OperationsFactory::OperationsFactory(const std::string& plugin,
                                     std::shared_ptr<NgraphNodes> nodes) {
    OperationsBase::sPluginType = plugin;
    OperationsBase::mNgraphNodes = nodes;
    ALOGV("%s Constructed", __func__);
}
OperationsFactory::~OperationsFactory() {
    OperationsBase::mNgraphNodes.reset();
    ALOGV("%s Destructed & reset", __func__);
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
        case OperationType::RESHAPE:
            mOperationsMap[type] = std::make_shared<Reshape>(model);
            break;
        default:
            return nullptr;
    }
    return mOperationsMap[type];
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android