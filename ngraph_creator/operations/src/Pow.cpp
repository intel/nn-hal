#include <Pow.hpp>
#define LOG_TAG "Pow"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Pow::Pow(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Pow::validate() {
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Pow::createNode() {
    // Creating input nodes
    auto base = getInputNode(0);
    auto exponent = getInputNode(1);

    auto outputNode = std::make_shared<ngraph::opset3::Power>(base, exponent,
                                                              ngraph::op::AutoBroadcastType::NUMPY);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
