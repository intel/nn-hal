#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class LogSoftmax : public OperationsBase {
public:
    LogSoftmax(int operationIndex);
    std::shared_ptr<ov::Node> createNode() override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
