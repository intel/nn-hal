#include <NgraphNetworkCreator.hpp>
#define LOG_TAG "NgraphNetworkCreator"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

NgraphNetworkCreator::NgraphNetworkCreator(const Model& model, const std::string& plugin)
    : mModel(model),
      mNgraphNodes(std::make_shared<NgraphNodes>(mModel.operands.size())),
      mOpFctryInst(plugin, mNgraphNodes) {
    ALOGD("NgraphNetworkCreator Constructed");
}

void NgraphNetworkCreator::createInputParams() {
    for (auto i : mModel.inputIndexes) {
        std::shared_ptr<ngraph::opset3::Parameter> inputParam;
        auto& origDims = mModel.operands[i].dimensions;
        std::vector<size_t> dims(origDims.begin(), origDims.end());
        if (dims.size() == 3) {
            ALOGD("createInputParams converting operand %d to 4D", i);
            dims.insert(dims.begin(), 1);
        }
        switch (mModel.operands[i].type) {
            case OperandType::FLOAT32:
            case OperandType::TENSOR_FLOAT32:
                inputParam = std::make_shared<ngraph::opset3::Parameter>(
                    ngraph::element::f32, ngraph::Shape(dims.begin(), dims.end()));
                ALOGD("createInputParams created inputIndex %d, type %d", i,
                      mModel.operands[i].type);
                break;
            default:
                ALOGE("createInputParams Failure at inputIndex %d, type %d", i,
                      mModel.operands[i].type);
                inputParam = nullptr;
        }
        mNgraphNodes->addInputParam(inputParam);
        mNgraphNodes->setOperationOutput(i, inputParam);
    }
}

bool NgraphNetworkCreator::validateOperations() {
    for (const auto& operation : mModel.operations) {
        if (!mOpFctryInst.getOperation(operation.type, mModel)->validate(operation)) return false;
    }
    return true;
}

bool NgraphNetworkCreator::initializeModel() {
    int index = 0;
    createInputParams();
    for (const auto& operation : mModel.operations) {
        auto op = mOpFctryInst.getOperation(operation.type, mModel);
        if (op == nullptr) {
            ALOGD("initializeModel Failure at type %d", operation.type);
            return false;
        }
        mNgraphNodes->setOperationOutput(
            operation.outputs[0],  // TODO: Handle multiple Outputs(eg.LSTM). Assumption here : each
                                   // node has only 1 output.
            op->createNodeForPlugin(operation));
    }
    ALOGD("initializeModel Success");
    return true;
}

const std::string& NgraphNetworkCreator::getNodeName(uint32_t index) {
    return mNgraphNodes->getNodeName(index);
}

std::shared_ptr<ngraph::Function> NgraphNetworkCreator::generateGraph() {
    for (auto i : mModel.outputIndexes) {
        mNgraphNodes->setResultNode(i);
    }
    return mNgraphNodes->generateGraph();
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
