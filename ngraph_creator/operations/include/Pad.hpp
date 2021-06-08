#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class Pad : public OperationsBase {
public:
    Pad(int operationIndex);
    bool validate() override;
    std::shared_ptr<ngraph::Node> createNode() override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
