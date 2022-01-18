#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class Bidirectional_Sequence_RNN : public OperationsBase {
public:
    Bidirectional_Sequence_RNN(int operationIndex);
    bool validate() override;
    std::shared_ptr<ngraph::Node> createNode() override;
    void connectOperationToGraph() override;
    bool isValidInputTensor(uint32_t inputIndex);
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android