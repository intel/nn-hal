//#define LOG_NDEBUG 0
#include <Argmin.hpp>
#define LOG_TAG "Argmin"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Argmin::Argmin(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Argmin::validate() {
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_INT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_INT32)) {
        return false;
    }
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Argmin::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;

    input = getInputNode(0);

    int32_t axis = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);
    ALOGD("createNode axis %d", axis);

    auto k_node = createConstNode(ngraph::element::i32, {}, convertToVector(1));

    auto topk = std::make_shared<ngraph::opset3::TopK>(
        input, k_node, axis, ngraph::opset3::TopK::Mode::MIN, ngraph::opset3::TopK::SortType::NONE);

    const auto axis_to_remove =
        createConstNode(ngraph::element::u32, {}, convertToVector(topk->get_axis()));
    auto outputNode = std::make_shared<ngraph::opset3::Squeeze>(topk->output(1), axis_to_remove);

    return outputNode;
}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
