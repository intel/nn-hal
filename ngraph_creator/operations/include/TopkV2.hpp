#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class TopkV2 : public OperationsBase {
public:
    TopkV2(int operationIndex);
    std::shared_ptr<ov::Node> createNode() override;
    void connectOperationToGraph() override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
