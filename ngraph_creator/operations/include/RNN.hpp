#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class RNN : public OperationsBase {
public:
    RNN(int operationIndex);
    bool validate() override;
    std::shared_ptr<ngraph::Node> createNode() override;
    void connectOperationToGraph() override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
