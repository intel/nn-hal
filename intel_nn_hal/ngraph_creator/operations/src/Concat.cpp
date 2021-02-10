#include <Concat.hpp>
#define LOG_TAG "Concat"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

// TODO: should use NNAPI_Utils:: GetConstOperand, ParseOperationInput
int GetConstOperand(const Model& model, uint32_t index) {
    const auto op = model.operands[index];
    if (op.lifetime == OperandLifeTime::CONSTANT_COPY) {
        if (op.location.poolIndex != 0) {
            ALOGE("CONSTANT_COPY expects poolIndex to be 0");
            // nnAssert(false);
        }
        ALOGD("operand lifetime OperandLifeTime::CONSTANT_COPY");
        return model.operandValues[op.location.offset];
    }
    ALOGE("FIX ME : Return unknown");
    return 0;
}
int ParseOperationInput(const Model& model, const Operation& operation, uint32_t index) {
    uint32_t inputIndex = operation.inputs[index];
    const auto operand = model.operands[inputIndex];
    return GetConstOperand(model, inputIndex);
}

Concat::Concat(const Model& model) : OperationsBase(model) {}

bool Concat::validate(const Operation& op) { return true; }

std::shared_ptr<ngraph::Node> Concat::createNode(const Operation& operation) {
    auto n = operation.inputs.size() - 1;
    std::vector<uint32_t> axisMap = {2, 3, 1};  // NCHW = axisMap[NHWC]
    auto axis = axisMap[ParseOperationInput(mModel, operation, n)];
    std::vector<std::shared_ptr<ngraph::Node>> inputs;
    ALOGD("createNode n %d, axis %d", n, axis);
    for (int i = 0; i < n; i++) {
        auto inputIndex = operation.inputs[i];
        auto inputOp = mNgraphNodes->getOperationOutput(inputIndex);
        const auto op = mModel.operands[inputIndex];
        ALOGD("createNode inputIndex %d, inputOp %s, lifetime %d", inputIndex,
              (inputOp ? "Valid" : "NULL"), op.lifetime);
        if (op.lifetime == OperandLifeTime::CONSTANT_COPY ||
            op.lifetime == OperandLifeTime::CONSTANT_REFERENCE ||
            op.lifetime ==
                OperandLifeTime::MODEL_INPUT)  // TODO: should use NNAPI_Utils::isConst || isInput
        {
            inputOp = transpose(NHWC_NCHW, inputOp);
        }
        inputs.push_back(inputOp);
    }

    return std::make_shared<ngraph::opset3::Concat>(inputs, axis);
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
