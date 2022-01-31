#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class Quantize : public OperationsBase {
public:
    Quantize(int operationIndex);
    std::shared_ptr<ngraph::Node> createNode() override;
    void connectOperationToGraph() override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
