#include <Concat.hpp>
#undef LOG_TAG
#define LOG_TAG "Concat"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Concat::Concat(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Concat::validate() {
    // check concatenation axis
    auto n = sModelInfo->getOperationInputsSize(mNnapiOperationIndex) -
             1;  // 0 ~ n-1: The list of n input tensors
    for (size_t i = 0; i < n; i++) {
        if (!isValidInputTensor(i)) {
            ALOGE("%s Invalid dimensions for input", __func__);
            return false;
        }
    }
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ov::Node> Concat::createNode() {
    auto n = sModelInfo->getOperationInputsSize(mNnapiOperationIndex) -
             1;  // 0 ~ n-1: The list of n input tensors
    auto axis = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex,
                                                          n);  // n: concatenation axis
    // TODO: Axis is 3 since data is in NHWC but due to optimization
    // data is in NCHW format
    // WA to fix Axis to 1
    axis = 1;
    std::vector<ov::Output<ov::Node>> inputs;
    ALOGD("createNode n %lu, axis %d", n, axis);
    for (size_t i = 0; i < n; i++) {
        auto inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, i);
        auto inputOp = getInputNode(i);
        const auto op = sModelInfo->getOperand(inputIndex);
        ALOGD("createNode inputIndex %d, lifetime %d", inputIndex, op.lifetime);
        inputs.push_back(inputOp);
    }

    std::shared_ptr<ov::Node> outputNode = std::make_shared<ov::opset3::Concat>(inputs, axis);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
