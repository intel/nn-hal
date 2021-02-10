#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
// Add is supposed to create an Add Node based on the arguments/parameters.
class Add : public OperationsBase {
public:
    Add(const Model& model);
    bool validate(const Operation& op) override;
    std::shared_ptr<ngraph::Node> createNode(const Operation& operation) override;
    std::shared_ptr<ngraph::Node> createNodeForPlugin(const Operation& operation) override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android