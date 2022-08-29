#include <Tanh.hpp>
#undef LOG_TAG
#define LOG_TAG "Tanh"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Tanh::Tanh(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Tanh::validate() {
    const auto inputRank = getInputOperandDimensions(0).size();
    if ( !isValidInputTensor(0) || inputRank > 4 ) {
         ALOGE("%s Empty  or Invalid dimensions size for input", __func__);
         return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Tanh::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;

    input = getInputNode(0);

    std::shared_ptr<ngraph::Node> outputNode;

    outputNode = std::make_shared<ngraph::opset3::Tanh>(input);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
