#include <PRelu.hpp>
#undef LOG_TAG
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

    const auto& baseDims = getInputOperandDimensions(0);
    const auto& alphaDims = getInputOperandDimensions(1);
    const auto& baseRank = baseDims.size();
    const auto& alphaRank = alphaDims.size();
    // TODO: openvino only supports broadcasting alpha rank/value to base rank/value. If alpha
    // rank/value is greater than base rank/value, base rank/value should be broadcasted to alpha
    // rank/value (which is not supported in openvino 2021.4)
    if (alphaRank > baseRank) return false;

    if (alphaRank == baseRank) {
        for (uint32_t i = 0; i < alphaRank; i++) {
            if (alphaDims[i] > baseDims[i]) return false;
        }
    }

    return true;
}

std::shared_ptr<ov::Node> PRelu::createNode() {
    // Creating input nodes
    auto base = getInputNode(0);
    auto alpha = getInputNode(1);

    auto outputNode = std::make_shared<ov::opset3::PRelu>(base, alpha);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
