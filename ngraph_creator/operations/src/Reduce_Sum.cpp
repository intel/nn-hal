#include <Reduce_Sum.hpp>
#define LOG_TAG "Reduce_Sum"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Reduce_Sum::Reduce_Sum(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Reduce_Sum::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) return false;

    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_INT32)) return false;

    // TODO: Remove this condition, Issue with "CNNNetworkImpl" in OpenVINO 2021.2, fixed in 2021.4
    // Issue is with input of shape [1, 1, 1] => Reduce_Sum is converted to Reshape using
    // ConvertReduce transformation and produces empty constant with [0] to reshape input [1, 1, 1]
    // to [] shape
    const auto dims = getInputOperandDimensions(0);
    bool isDimsOne = true;
    if (std::count(std::begin(dims), std::end(dims), dims.front()) == dims.size()) {
        if (dims[0] == 1) return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Reduce_Sum::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);
    auto reduction_axes = getInputNode(1);
    auto keep_dims = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 2);

    auto outputNode = std::make_shared<ngraph::opset3::ReduceSum>(input, reduction_axes, keep_dims);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android