#include <DepthToSpace.hpp>
#undef LOG_TAG
#define LOG_TAG "DepthToSpace"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

DepthToSpace::DepthToSpace(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ngraph::Node> DepthToSpace::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;
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

    std::shared_ptr<ngraph::Node> outputNode;

    outputNode = std::make_shared<ngraph::opset3::DepthToSpace>(
        input, ngraph::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, block_size);

    if (!useNchw) outputNode = transpose(NCHW_NHWC, outputNode);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
