#pragma once

#include <Driver.h>
#include <log/log.h>

#define OP_INPUT_IDX_CONV 0
static const std::string INVALID_STRING("Invalid Node");

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

// TODO: should use NNAPI_Utils:: GetConstOperand, ParseOperationInput
static int GetConstOperand(const Model& model, uint32_t index) {
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
static int ParseOperationInput(const Model& model, const Operation& operation, uint32_t index) {
    uint32_t inputIndex = operation.inputs[index];
    const auto operand = model.operands[inputIndex];
    return GetConstOperand(model, inputIndex);
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android