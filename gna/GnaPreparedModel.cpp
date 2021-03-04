#define LOG_TAG "GnaPreparedModel"

#include "GnaPreparedModel.h"
#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>
#include <fstream>
#include <thread>
#include "ValidateHal.h"
#include "utils.h"

using namespace android::nn;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

void GnaPreparedModel::deinitialize() {
    ALOGV("Entering %s", __func__);
    delete mEnginePtr;
    mEnginePtr = nullptr;
    ALOGI("free engine");
    ALOGV("Exiting %s", __func__);
}

bool GnaPreparedModel::initialize(const Model &model) {
    ALOGV("Entering %s", __func__);
    bool success = false;

    // NgraphNetworkCreator ngc(mModel, "CPU");
    mNgc->initializeModel();  // NgraphNetworkCreator
    auto ngraph_function = mNgc->generateGraph();
    InferenceEngine::CNNNetwork ngraph_net = InferenceEngine::CNNNetwork(ngraph_function);
    ngraph_net.serialize("/data/vendor/neuralnetworks/ngraph_ir.xml",
                         "/data/vendor/neuralnetworks/ngraph_ir.bin");

    // Check operation supoorted or not, user may not call getOpertionSupported()
    for (const auto &operation : mModelInfo->mModel.operations) {
        success = isOperationSupported(operation, mModelInfo->mModel);
        dumpOperationSupport(operation, success);
        if (!success) {
            ALOGE("Unsupported operation in initialize()");
            return false;
        }
    }

    success = mModelInfo->setRunTimePoolInfosFromHidlMemories(mModelInfo->mModel.pools);
    if (!success) {
        ALOGE("setRunTimePoolInfosFromHidlMemories failed.");
        return false;
    }

    if (!mModelInfo->initRuntimeInfo()) {
        ALOGE("Failed to initialize Model runtime parameters!!");
        return false;
    }

    ALOGI("initialize ExecuteNetwork for device %s", mTargetDevice.c_str());
    mEnginePtr = new ExecuteNetwork(ngraph_net, mTargetDevice);
    mEnginePtr->prepareInput();
    mEnginePtr->loadNetwork(ngraph_net);
    ALOGV("Exiting %s", __func__);
    return true;
}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
