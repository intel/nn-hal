#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class StridedSlice : public OperationsBase {
public:
    StridedSlice(int operationIndex);
    bool validate() override;
    std::shared_ptr<ov::Node> createNode() override;
    std::vector<int64_t> getMaskBits(int32_t maskValue, size_t vec_size);
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
