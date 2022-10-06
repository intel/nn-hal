#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class MaxPool2d : public OperationsBase {
public:
    MaxPool2d(int operationIndex);
    std::shared_ptr<ov::Node> createNode() override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
