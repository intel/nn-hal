#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class Add : public OperationsBase {
public:
    Add(const Operation& op);
    bool validate() override;
    std::shared_ptr<ngraph::Node> createNode() override;
    std::shared_ptr<ngraph::Node> createNodeForPlugin() override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android