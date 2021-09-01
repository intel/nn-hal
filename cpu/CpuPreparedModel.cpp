#include "CpuPreparedModel.h"
#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>
#include <fstream>
#include <thread>
#include "ValidateHal.h"
#include "utils.h"

#define LOG_TAG "CpuPreparedModel"

using namespace android::nn;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

void CpuPreparedModel::deinitialize() {
    ALOGV("Entering %s", __func__);
    mModelInfo->unmapRuntimeMemPools();

    ALOGV("Exiting %s", __func__);
}

bool CpuPreparedModel::initialize() {
    ALOGV("Entering %s", __func__);
    if (!mModelInfo->initRuntimeInfo()) {
        ALOGE("Failed to initialize Model runtime parameters!!");
        return false;
    }
    mNgc = std::make_shared<NgraphNetworkCreator>(mModelInfo, mTargetDevice);

    if (!mNgc->validateOperations()) return false;
    ALOGI("Generating IR Graph");
    auto ngraph_function = mNgc->generateGraph();
    if (ngraph_function == nullptr) {
        ALOGE("%s ngraph generation failed", __func__);
        return false;
    }
    try {
        cnnNetworkPtr = std::make_shared<InferenceEngine::CNNNetwork>(ngraph_function);
#if __ANDROID__
        cnnNetworkPtr->serialize("/data/vendor/neuralnetworks/ngraph_ir.xml",
                                 "/data/vendor/neuralnetworks/ngraph_ir.bin");
#else
        cnnNetworkPtr->serialize("/tmp/ngraph_ir.xml", "/tmp/ngraph_ir.bin");
#endif
        mPlugin = std::make_shared<IENetwork>(cnnNetworkPtr);
        mPlugin->loadNetwork();
    } catch (const std::exception& ex) {
        ALOGE("%s Exception !!! %s", __func__, ex.what());
        return false;
    }

    ALOGV("Exiting %s", __func__);
    return true;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
