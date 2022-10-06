#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class UnidirectionalSequenceRNN : public OperationsBase {
public:
    UnidirectionalSequenceRNN(int operationIndex);
    void connectOperationToGraph() override;
    std::shared_ptr<ov::Node> createNode() override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
