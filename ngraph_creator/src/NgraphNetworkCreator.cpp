#include <NgraphNetworkCreator.hpp>
#define LOG_TAG "NgraphNetworkCreator"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

NgraphNetworkCreator::NgraphNetworkCreator(NnapiModelInfo* model, const std::string& plugin)
    : mModelInfo(model),
      mNgraphNodes(std::make_shared<NgraphNodes>(mModelInfo->getModel().operands.size())),
      mOpFctryInst(plugin, mNgraphNodes) {
    ALOGD("NgraphNetworkCreator Constructed");
}

void NgraphNetworkCreator::createInputParams() {
    for (auto i : mModelInfo-> getModelInputIndexes()) {
        std::shared_ptr<ngraph::opset3::Parameter> inputParam;
        auto& origDims = mModelInfo->getOperand(i).dimensions;
        std::vector<size_t> dims(origDims.begin(), origDims.end());
        if (dims.size() == 3) {  // TODO:Handle other dims size too
            ALOGI("createInputParams converting operand %d to 4D", i);
            dims.insert(dims.begin(), 1);
        }
        switch (mModelInfo->getOperand(i).type) {
            case OperandType::FLOAT32:
            case OperandType::TENSOR_FLOAT32:
                inputParam = std::make_shared<ngraph::opset3::Parameter>(
                    ngraph::element::f32, ngraph::Shape(dims.begin(), dims.end()));
                ALOGV("createInputParams created inputIndex %d, type %d", i,
                      mModelInfo->getOperand(i).type);
                break;
            default:
                ALOGE("createInputParams Failure at inputIndex %d, type %d", i,
                      mModelInfo->getOperand(i).type);
                inputParam = nullptr;
        }
        mNgraphNodes->addInputParam(i, inputParam);
        mNgraphNodes->setOperationOutput(i, inputParam);
    }
}

bool NgraphNetworkCreator::validateOperations() {
    for (const auto& operation : mModelInfo->getOperations()) {
        if (!mOpFctryInst.getOperation(operation.type, mModelInfo)->validate(operation)) return false;
    }
    return true;
}

bool NgraphNetworkCreator::initializeModel() {
    int index = 0;
    createInputParams();
    for (const auto& operation : mModelInfo->getOperations()) {
        auto op = mOpFctryInst.getOperation(operation.type,  mModelInfo);
        if (op == nullptr) {
            ALOGE("initializeModel Failure at type %d", operation.type);
            return false;
        }
        op->connectOperationToGraph(operation);
    }
    ALOGD("initializeModel Success");
    return true;
}

const std::string& NgraphNetworkCreator::getNodeName(uint32_t index) {
    ALOGD("getNodeName %d", index);
    return mNgraphNodes->getNodeName(index);
}

std::shared_ptr<ngraph::Function> NgraphNetworkCreator::generateGraph() {
    for (auto i : mModelInfo->getModelOutputIndexes()) {
        ALOGD("setResultNode %d", i);
        mNgraphNodes->setResultNode(i);
    }
    return mNgraphNodes->generateGraph();
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
