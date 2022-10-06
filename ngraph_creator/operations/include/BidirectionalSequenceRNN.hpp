#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class BidirectionalSequenceRNN : public OperationsBase {
public:
    BidirectionalSequenceRNN(int operationIndex);
    std::shared_ptr<ov::Node> createNode() override;
    void connectOperationToGraph() override;
    bool isValidInputTensor(uint32_t inputIndex);
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
