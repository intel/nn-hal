#include <Concat.hpp>
#define LOG_TAG "Concat"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Concat::Concat(const Model& model) : OperationsBase(model) {}

bool Concat::validate(const Operation& op) { return true; }

std::shared_ptr<ngraph::Node> Concat::createNode(const Operation& operation) {
    auto n = operation.inputs.size() - 1;
    std::vector<uint32_t> axisMap = {2, 3, 1};  // NCHW = axisMap[NHWC]
    auto axis = axisMap[ParseOperationInput(mModel, operation, n)];
    std::vector<ngraph::Output<ngraph::Node>> inputs;
    ALOGD("createNode n %d, axis %d", n, axis);
    for (int i = 0; i < n; i++) {
        auto inputIndex = operation.inputs[i];
        auto inputOp = mNgraphNodes->getOperationOutput(inputIndex);
        const auto op = mModel.operands[inputIndex];
        ALOGD("createNode inputIndex %d, lifetime %d", inputIndex, op.lifetime);
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
