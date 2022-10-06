#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class LSTM : public OperationsBase {
public:
    LSTM(int operationIndex);
    bool validate() override;
    std::shared_ptr<ov::Node> createNode() override;
    void connectOperationToGraph() override;

    std::shared_ptr<ov::Node> add(const ov::Output<ov::Node>& lhs, const ov::Output<ov::Node>& rhs);
    std::shared_ptr<ov::Node> sub(const ov::Output<ov::Node>& lhs, const ov::Output<ov::Node>& rhs);
    std::shared_ptr<ov::Node> mul(const ov::Output<ov::Node>& lhs, const ov::Output<ov::Node>& rhs);
    std::shared_ptr<ov::Node> matMul(const ov::Output<ov::Node>& lhs,
                                     const ov::Output<ov::Node>& rhs, bool transpose_lhs,
                                     bool transpose_rhs);
    std::shared_ptr<ov::Node> clip(const ov::Output<ov::Node>& data, float m_clip) const;
    std::shared_ptr<ov::Node> applyActivation(const std::shared_ptr<ov::Node>& arg,
                                              int activationFn) const;
    std::shared_ptr<ov::Node> LayerNorm(const ov::Output<ov::Node>& input,
                                        const std::shared_ptr<ov::Node>& normalizedweights,
                                        const std::shared_ptr<ov::Node>& bias);

    bool isValidInputTensor(uint32_t inputIndex);
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
