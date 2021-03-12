//#define LOG_NDEBUG 0
#include <NgraphNetworkCreator.hpp>
#define LOG_TAG "NgraphNetworkCreator"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

NgraphNetworkCreator::NgraphNetworkCreator(const Model& model, const std::string& plugin)
    : mModel(model),
      mNgraphNodes(
          std::make_shared<NgraphNodes>(mModel.operands.size(), mModel.outputIndexes.size())),
      mOpFctryInst(plugin, mModel, mNgraphNodes) {
    mOperations.resize(mModel.operations.size());
    int operationIndex = 0;
    for (const auto& operation : mModel.operations) {
        auto opInstance = mOpFctryInst.getOperation(operation);
        if (opInstance == nullptr) {
            ALOGE("%s Unsupported Operation type %d", __func__, operation.type);
        }
        mOperations[operationIndex++] = opInstance;
    }
    ALOGV("%s Constructed", __func__);
}

NgraphNetworkCreator::~NgraphNetworkCreator() { ALOGV("%s Destructed", __func__); }

void NgraphNetworkCreator::createInputParams() {
    for (auto i : mModel.inputIndexes) {
        std::shared_ptr<ngraph::opset3::Parameter> inputParam;
        auto& origDims = mModel.operands[i].dimensions;
        std::vector<size_t> dims(origDims.begin(), origDims.end());
        ALOGI("createInputParams operand %d dims.size(%d)", i, dims.size());
        switch (mModel.operands[i].type) {
            case OperandType::FLOAT32:
            case OperandType::TENSOR_FLOAT32:
                inputParam = std::make_shared<ngraph::opset3::Parameter>(
                    ngraph::element::f32, ngraph::Shape(dims.begin(), dims.end()));
                ALOGV("createInputParams created inputIndex %d, type %d", i,
                      mModel.operands[i].type);
                break;
            default:
                ALOGE("createInputParams Failure at inputIndex %d, type %d", i,
                      mModel.operands[i].type);
                inputParam = nullptr;
        }
        mNgraphNodes->addInputParam(inputParam);
        mNgraphNodes->setOutputAtOperandIndex(i, inputParam);
    }
}

bool NgraphNetworkCreator::validateOperations() {
    for (int i = 0; i < mModel.operations.size(); i++) {
        if (!mOperations[i] || !mOperations[i]->validate()) return false;
    }
    return true;
}

bool NgraphNetworkCreator::initializeModel() {
    ALOGV("%s Called", __func__);
    createInputParams();
    for (int i = 0; i < mModel.operations.size(); i++) {
        if (mOperations[i] == nullptr) {
            ALOGE("initializeModel Failure at type %d", mModel.operations[i].type);
            return false;
        }
        try {
            mOperations[i]->connectOperationToGraph();
        } catch (const std::exception& ex) {
            ALOGE("%s Exception !!! %s", __func__, ex.what());
            return false;
        }
    }
    ALOGD("initializeModel Success");
    return true;
}

const std::string& NgraphNetworkCreator::getNodeName(uint32_t index) {
    ALOGV("getNodeName %d", index);
    return mNgraphNodes->getNodeName(index);
}

std::shared_ptr<ngraph::Function> NgraphNetworkCreator::generateGraph() {
    ALOGV("%s Called", __func__);
    std::shared_ptr<ngraph::Function> ret;
    try {
        ret = mNgraphNodes->generateGraph();
    } catch (const std::exception& ex) {
        ALOGE("%s Exception !!! %s", __func__, ex.what());
    }
    return ret;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
