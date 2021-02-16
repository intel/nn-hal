#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class Reshape : public OperationsBase {
public:
    Reshape(const Model& model);
    bool validate(const Operation& op) override;
    std::shared_ptr<ngraph::Node> createNode(const Operation& operation) override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android