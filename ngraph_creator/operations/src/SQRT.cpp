#include <SQRT.hpp>
#define LOG_TAG "SQRT"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

SQRT::SQRT(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool SQRT::validate() {
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> SQRT::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);

    auto outputNode = std::make_shared<ngraph::opset3::Sqrt>(input);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
