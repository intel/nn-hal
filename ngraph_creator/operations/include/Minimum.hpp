#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class Minimum : public OperationsBase {
public:
    Minimum(int operationIndex);
    std::shared_ptr<ngraph::Node> createNode() override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
