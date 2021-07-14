#include <OperationsBase.hpp>
#define LOG_TAG "OperationsBase"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

IntelDeviceType OperationsBase::sPluginType;
std::shared_ptr<NnapiModelInfo> OperationsBase::sModelInfo;

std::shared_ptr<ngraph::Node> OperationsBase::transpose(ConversionType type,
                                                        ngraph::Output<ngraph::Node> input) {
    ngraph::AxisVector order;
    switch (type) {
        case NHWC_NCHW:
            order = {0, 3, 1, 2};
            break;
        case NCHW_NHWC:
            order = {0, 2, 3, 1};
            break;
        case IHWO_OIHW:
            order = {3, 0, 1, 2};
            break;
        case OHWI_OIHW:
            order = {0, 3, 1, 2};
            break;
        case NHWC_CWHN:
            order = {3, 2, 1, 0};
            break;
        case CWHN_NHWC:
            order = {3, 2, 1, 0};
            break;
        case NHC_NCH:
            order = {0, 2, 1};
            break;
        case NCH_NHC:
            order = {0, 1, 2};
            break;
        case NC_CN:
            order = {1, 0};
            break;
    }
    const auto order_node =
        ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{order.size()}, order);
    return std::make_shared<ngraph::opset3::Transpose>(input, order_node);
}

std::shared_ptr<ngraph::Node> OperationsBase::toNCHW(size_t inputIndex, size_t outputIndex) {
    auto inNode = mNgraphNodes->getOperationOutput(inputIndex).get_node_shared_ptr();
    if (mNgraphNodes->isForcedNchw(inputIndex))
        return inNode;
    else {
        mNgraphNodes->setForcedNchw(outputIndex, true);
        return transpose(NHWC_NCHW, inNode);
    }
}

// override createNodeForPlugin in case sPluginType specific implementation is required
std::shared_ptr<ngraph::Node> OperationsBase::createNodeForPlugin() { return createNode(); }

// override connectOperationToGraph in case Operation has multiple outputs
void OperationsBase::connectOperationToGraph() {
    auto outputNode = createNodeForPlugin();
    const auto op = sModelInfo->getOperand(mDefaultOutputIndex);
    if (op.type == OperandType::TENSOR_QUANT8_ASYMM) {
        outputNode = QuantizeNode(outputNode, mDefaultOutputIndex, ngraph::element::u8);
    }
    if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
        addResultNode(mDefaultOutputIndex, outputNode);
    }
    mNgraphNodes->setOutputAtOperandIndex(mDefaultOutputIndex, outputNode->get_default_output());
}

void OperationsBase::addResultNode(size_t index, std::shared_ptr<ngraph::Node> resultNode) {
    mNgraphNodes->setResultNode(index, resultNode);
}

OperationsBase::OperationsBase(int operationIndex) : mNnapiOperationIndex(operationIndex) {
    mDefaultOutputIndex = 0;
}

void OperationsBase::setNgraphNodes(std::shared_ptr<NgraphNodes> nodes) { mNgraphNodes = nodes; }

bool OperationsBase::validate() { return true; }

bool OperationsBase::validateForPlugin() {
    // Only validates default input(initialized to 0)
    // All other validations to be done at each operation's validate
    if (!isValidInputTensor(mDefaultInputIndex)) {
        ALOGE("%s Invalid dimensions for input", __func__);
        return false;
    }
    return validate();
}

bool OperationsBase::checkOperandType(uint32_t operandIndex, const int32_t expectedOperandType,
                                      const std::string& strLogInfo) {
    const auto operandType = (int32_t)sModelInfo->getOperandType(operandIndex);
    if (operandType != expectedOperandType) {
        ALOGE("OperationIndex %d %s Index %d type %d invalid", mNnapiOperationIndex,
              strLogInfo.c_str(), operandIndex, operandType);
        return false;
    }
    ALOGV("OperationIndex %d %s Index %d type %d PASSED", mNnapiOperationIndex, strLogInfo.c_str(),
          operandIndex, operandType);
    return true;
}
bool OperationsBase::checkOutputOperandType(uint32_t index, const int32_t expectedOperandType) {
    const auto& operandIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, index);
    return checkOperandType(operandIndex, expectedOperandType, "Output");
}
bool OperationsBase::checkInputOperandType(uint32_t index, const int32_t expectedOperandType) {
    const auto& operandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, index);
    return checkOperandType(operandIndex, expectedOperandType, "Input");
}
const vec<uint32_t> OperationsBase::getInputOperandDimensions(uint32_t inputIndex) {
    const auto& operandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, inputIndex);
    const auto& operand = sModelInfo->getOperand(operandIndex);
    return operand.dimensions;
}

bool OperationsBase::isValidInputTensor(uint32_t inputIndex) {
    size_t size = 1;
    const auto& dims = getInputOperandDimensions(inputIndex);
    ALOGV("%s dims.size(%d)", __func__, dims.size());
    if (dims.empty()) return false;

    for (auto d : dims) size *= d;
    if (size == 0) return false;

    return true;
}

std::shared_ptr<ngraph::Node> OperationsBase::QuantizeNode(std::shared_ptr<ngraph::Node> input,
                                                           size_t index,
                                                           ngraph::element::Type quantizeType) {
    auto floatElementType = ngraph::element::f32;
    auto intElementType = ngraph::element::i32;

    float inputScale = sModelInfo->getOperandScale(index);
    int inputZeroPoint = sModelInfo->getOperandZeroPoint(index);

    auto scale = createConstNode(floatElementType, {}, convertToVector(inputScale));
    auto zeroPoint = createConstNode(intElementType, {}, convertToVector(inputZeroPoint));

    if (input->get_element_type() != ngraph::element::f32)
        input = std::make_shared<ngraph::opset3::Convert>(input, floatElementType);
    auto div = std::make_shared<ngraph::opset3::Divide>(input, scale);
    ngraph::op::v5::Round::RoundMode mode = ngraph::op::v5::Round::RoundMode::HALF_TO_EVEN;
    auto round = std::make_shared<ngraph::op::v5::Round>(div, mode);
    auto convertRound = std::make_shared<ngraph::opset3::Convert>(round, ngraph::element::i32);
    auto sum = std::make_shared<ngraph::opset3::Add>(convertRound, zeroPoint);
    auto data = make_shared<ngraph::opset3::Clamp>(sum, 0, 255);

    auto outputNode = std::make_shared<ngraph::opset3::Convert>(data, quantizeType);

    return outputNode;
}

std::shared_ptr<ngraph::Node> OperationsBase::DequantizeNode(std::shared_ptr<ngraph::Node> input,
                                                             size_t index,
                                                             ngraph::element::Type dequantizeType) {
    auto floatElementType = ngraph::element::f32;
    auto intElementType = ngraph::element::i32;

    float inputScale = sModelInfo->getOperandScale(index);
    int inputZeroPoint = sModelInfo->getOperandZeroPoint(index);

    auto scale = createConstNode(floatElementType, {}, convertToVector(inputScale));
    auto zeroPoint = createConstNode(intElementType, {}, convertToVector(inputZeroPoint));

    if (input->get_element_type() != ngraph::element::i32)
        input = std::make_shared<ngraph::opset3::Convert>(input, intElementType);
    auto diff = std::make_shared<ngraph::opset3::Subtract>(input, zeroPoint);
    auto convertDiff = std::make_shared<ngraph::opset3::Convert>(diff, floatElementType);
    auto mul = std::make_shared<ngraph::opset3::Multiply>(convertDiff, scale);

    auto outputNode = std::make_shared<ngraph::opset3::Convert>(mul, dequantizeType);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
