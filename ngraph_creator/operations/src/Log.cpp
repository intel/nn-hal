#include <Log.hpp>
#undef LOG_TAG
#define LOG_TAG "Log"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Log::Log(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> Log::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);

    auto outputNode = std::make_shared<ov::opset3::Log>(input);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
