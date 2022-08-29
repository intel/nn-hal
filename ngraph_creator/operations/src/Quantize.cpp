#include <Quantize.hpp>
#undef LOG_TAG
#define LOG_TAG "Quantize"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Quantize::Quantize(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Quantize::validate() {
    if ( !isValidInputTensor(0)) {
         ALOGE("%s Empty  or Invalid dimensions size for input", __func__);
         return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

void Quantize::connectOperationToGraph() { createNode(); }

std::shared_ptr<ngraph::Node> Quantize::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);
    const auto& outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
    auto outputNode = QuantizeNode(input, outputIndex, ngraph::element::u8);

    mNgraphNodes->setOutputAtOperandIndex(outputIndex, outputNode);
    const auto op = sModelInfo->getOperand(outputIndex);
    if (op.lifetime == OperandLifeTime::SUBGRAPH_OUTPUT) {
        addResultNode(mDefaultOutputIndex, outputNode);
    }

    return nullptr;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
