#include <PRelu.hpp>
#define LOG_TAG "PRelu"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

PRelu::PRelu(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool PRelu::validate() {
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> PRelu::createNode() {
    // Creating input nodes
    auto base = getInputNode(0);
    auto alpha = getInputNode(1);

    auto outputNode = std::make_shared<ngraph::opset3::PRelu>(base, alpha);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
