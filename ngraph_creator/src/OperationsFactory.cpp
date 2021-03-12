#include <OperationsFactory.hpp>
#define LOG_TAG "OperationsFactory"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

OperationsFactory::OperationsFactory(const std::string& plugin, std::shared_ptr<NnapiModelInfo> modelInfo,
                                     std::shared_ptr<NgraphNodes> nodes) {
    OperationsBase::sPluginType = plugin;
    OperationsBase::sModelInfo = modelInfo;
    OperationsBase::mNgraphNodes = nodes;
    ALOGV("%s Constructed", __func__);
}
OperationsFactory::~OperationsFactory() {
    OperationsBase::mNgraphNodes.reset();
    ALOGV("%s Destructed & reset", __func__);
}
std::shared_ptr<OperationsBase> OperationsFactory::getOperation(int operationIndex, const OperationType& operationType) {
    switch (operationType) {
        case OperationType::ADD:
            return std::make_shared<Add>(operationIndex);
        case OperationType::CONCATENATION:
            return std::make_shared<Concat>(operationIndex);
        case OperationType::RESHAPE:
            return std::make_shared<Reshape>(operationIndex);
        default:
            ALOGE("%s Cannot identify OperationType %d", __func__, operationType);
            break;
    }
    return nullptr;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android