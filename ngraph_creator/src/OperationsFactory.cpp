#include <OperationsFactory.hpp>
#define LOG_TAG "OperationsFactory"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

OperationsFactory::OperationsFactory(const std::string& plugin, Model& model,
                                     std::shared_ptr<NgraphNodes> nodes) {
    OperationsBase::sPluginType = plugin;
    OperationsBase::sModel = &model;
    OperationsBase::mNgraphNodes = nodes;
    ALOGV("%s Constructed", __func__);
}
OperationsFactory::~OperationsFactory() {
    OperationsBase::mNgraphNodes.reset();
    ALOGV("%s Destructed & reset", __func__);
}
std::shared_ptr<OperationsBase> OperationsFactory::getOperation(const Operation& op) {
    switch (op.type) {
        case OperationType::ADD:
            return std::make_shared<Add>(op);
        case OperationType::CONCATENATION:
            return std::make_shared<Concat>(op);
        case OperationType::RESHAPE:
            return std::make_shared<Reshape>(op);
        default:
            ALOGE("%s Cannot identify OperationType %d", __func__, op.type);
            break;
    }
    return nullptr;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android