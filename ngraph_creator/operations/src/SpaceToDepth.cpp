#include <SpaceToDepth.hpp>
#undef LOG_TAG
#define LOG_TAG "SpaceToDepth"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

SpaceToDepth::SpaceToDepth(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> SpaceToDepth::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input;
    bool useNchw = false;

    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);

    if (inputsSize == 3) {
        auto layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 2);
        if (layout) useNchw = true;
    }

    input = getInputNode(0);
    auto block_size = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 1);

    if (!useNchw)  // No conversion needed if useNchw set
        input = transpose(NHWC_NCHW, input);

    std::shared_ptr<ov::Node> outputNode;

    outputNode = std::make_shared<ov::opset3::SpaceToDepth>(
        input, ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, block_size);

    if (!useNchw) outputNode = transpose(NCHW_NHWC, outputNode);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
