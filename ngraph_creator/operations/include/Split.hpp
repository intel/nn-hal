#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class Split : public OperationsBase {
public:
    Split(int operationIndex);
    std::shared_ptr<ngraph::Node> createNode() override;
    void connectOperationToGraph() override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
