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
    mModelInfo->unmapRuntimeMemPools();

    ALOGV("Exiting %s", __func__);
}

bool GnaPreparedModel::initialize() {
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
    auto ngraph_net = std::make_shared<InferenceEngine::CNNNetwork>(ngraph_function);
#if __ANDROID__
    ngraph_net->serialize("/data/vendor/neuralnetworks/ngraph_ir.xml",
                          "/data/vendor/neuralnetworks/ngraph_ir.bin");
#else
    ngraph_net->serialize("/tmp/ngraph_ir.xml", "/tmp/ngraph_ir.bin");
#endif
    mPlugin = std::make_shared<IENetwork>(ngraph_net);
    mPlugin->loadNetwork();

    ALOGV("Exiting %s", __func__);
    return true;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
