#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class Add : public OperationsBase {
public:
    Add(int operationIndex);
    bool validate() override;
    std::shared_ptr<ov::Node> createNode() override;
    std::shared_ptr<ov::Node> createNodeForPlugin() override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
