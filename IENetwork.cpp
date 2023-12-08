#include "IENetwork.h"
#include "ie_common.h"

#include <android-base/logging.h>
#include <android/log.h>
#include <ie_blob.h>
#include <log/log.h>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>

#undef LOG_TAG
#define LOG_TAG "IENetwork"

namespace android::hardware::neuralnetworks::nnhal {

bool IENetwork::loadNetwork() {
    ALOGD("%s", __func__);

#if __ANDROID__
    ov::Core ie(std::string("/vendor/etc/openvino/plugins.xml"));
#else
    ov::Core ie(std::string("/usr/local/lib64/plugins.xml"));
#endif
    std::map<std::string, std::string> config;
    std::string deviceStr;
    switch (mTargetDevice) {
        case IntelDeviceType::GNA:
            deviceStr = "GNA";
            break;
        case IntelDeviceType::NPU:
            deviceStr = "NPU";
            break;
        case IntelDeviceType::CPU:
        default:
            deviceStr = "CPU";
            break;
    }

    ALOGD("Creating infer request for Intel Device Type : %s", deviceStr.c_str());
    if (mNetwork) {
        compiled_model = ie.compile_model(mNetwork, deviceStr);
        ALOGD("LoadNetwork is done....");

#if __ANDROID__
        ov::pass::Serialize serializer("/data/vendor/neuralnetworks/ngraph_ir.xml",
                                       "/data/vendor/neuralnetworks/ngraph_ir.bin");
        serializer.run_on_model(
            std::const_pointer_cast<ov::Model>(compiled_model.get_runtime_model()));
#else
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::Serialize>("/tmp/model.xml", "/tmp/model.bin");
        manager.run_passes(mNetwork);
#endif

        std::vector<ov::Output<ov::Node>> modelInput = mNetwork->inputs();
        mInferRequest = compiled_model.create_infer_request();
        ALOGD("CreateInfereRequest is done....");

    } else {
        ALOGE("Invalid Network pointer");
        return false;
    }

    return true;
}

// Need to be called before loadnetwork.. But not sure whether need to be called for
// all the inputs in case multiple input / output

ov::Tensor IENetwork::getBlob(const std::string& outName) {
    return mInferRequest.get_tensor(outName);
}

ov::Tensor IENetwork::getInputBlob(const std::size_t index) {
    return mInferRequest.get_input_tensor(index);
}
ov::Tensor IENetwork::getOutputBlob(const std::size_t index) {
    return mInferRequest.get_output_tensor(index);
}
void IENetwork::infer() {
    ALOGI("Infer Network\n");
    mInferRequest.start_async();
    mInferRequest.wait_for(std::chrono::milliseconds(10000));
    ALOGI("infer request completed");
}

}  // namespace android::hardware::neuralnetworks::nnhal
