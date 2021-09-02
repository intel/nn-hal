#include <Instance_Normalization.hpp>
#define LOG_TAG "Instance_Normalization"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Instance_Normalization::Instance_Normalization(int operationIndex)
    : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Instance_Normalization::validate() {
    ALOGV("%s Entering", __func__);
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        ALOGE("%s Output operand 0 is not of type FP32. Unsupported operation", __func__);
        return false;
    }

    // Check Input Type
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        ALOGE("%s Input operand 0 is not of type FP32. Unsupported operation", __func__);
        return false;
    }
    const auto inputRank = getInputOperandDimensions(0).size();
    if ((inputRank > 4) || (!isValidInputTensor(0))) {
        ALOGE("%s Invalid dimensions size for input(%lu)", __func__, inputRank);
        return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Instance_Normalization::createNode() {
    ALOGV("%s Entering", __func__);

    std::shared_ptr<ngraph::Node> inputNode;
    bool useNchw = false;
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    ALOGD("%s inputsSize %lu", __func__, inputsSize);

    // Read inputs
    inputNode = getInputNode(0);
    auto gamma = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 1);
    auto beta = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 2);
    auto epsilon = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 3);
    auto layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 4);
    if (layout) useNchw = true;

    if (!useNchw)  // No conversion needed if useNchw set
        inputNode = transpose(NHWC_NCHW, inputNode);

    // output[b, h, w, c] =   (input[b, h, w, c] - mean[b, c]) * gamma /
    //                                         sqrt(var[b, c] + epsilon) + beta
    // Instance Normalizatiom = MVN * gamma + beta
    bool normalize_variance = true;
    auto gammaNode = createConstNode(ngraph::element::f32, {1}, convertToVector(gamma));
    auto betaNode = createConstNode(ngraph::element::f32, {1}, convertToVector(beta));

    // Axis along which mean and variance is calculated
    std::vector<int32_t> axes{2, 3};
    std::shared_ptr<ngraph::Node> inputAxesNode = createConstNode(ngraph::element::i32, {2}, axes);
    std::shared_ptr<ngraph::Node> mvnNode = std::make_shared<ngraph::op::v6::MVN>(
        inputNode, inputAxesNode, normalize_variance, epsilon, ngraph::op::MVNEpsMode::INSIDE_SQRT);

    auto mulGamma = std::make_shared<ngraph::opset3::Multiply>(
        mvnNode, gammaNode, ngraph::op::AutoBroadcastType::NUMPY);
    std::shared_ptr<ngraph::Node> outputNode =
        std::make_shared<ngraph::opset3::Add>(mulGamma, betaNode);

    if (!useNchw) outputNode = transpose(NCHW_NHWC, outputNode);
    ALOGV("%s PASSED", __func__);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
