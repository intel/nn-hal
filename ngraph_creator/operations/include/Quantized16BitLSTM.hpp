#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class Quantized16BitLSTM : public OperationsBase {
public:
    Quantized16BitLSTM(int operationIndex);
    bool validate() override;
    std::shared_ptr<ngraph::Node> createNode() override;
    void connectOperationToGraph() override;

    std::shared_ptr<ngraph::Node> applyActivation(const std::shared_ptr<ngraph::Node>& arg,
                                                  int activationFn) const;
    std::shared_ptr<ngraph::Node> add(const ngraph::Output<ngraph::Node>& lhs,
                                      const ngraph::Output<ngraph::Node>& rhs);
    std::shared_ptr<ngraph::Node> sub(const ngraph::Output<ngraph::Node>& lhs,
                                      const ngraph::Output<ngraph::Node>& rhs);
    std::shared_ptr<ngraph::Node> mul(const ngraph::Output<ngraph::Node>& lhs,
                                      const ngraph::Output<ngraph::Node>& rhs);
    std::shared_ptr<ngraph::Node> matMul(const ngraph::Output<ngraph::Node>& lhs,
                                         const ngraph::Output<ngraph::Node>& rhs,
                                         bool transpose_lhs, bool transpose_rhs);
    std::shared_ptr<ngraph::Node> clip(const ngraph::Output<ngraph::Node>& data,
                                       float m_clip) const;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
