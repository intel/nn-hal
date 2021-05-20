#define LOG_TAG "IENetwork"
#include "IENetwork.h"
#include "ie_common.h"

#include <android-base/logging.h>
#include <android/log.h>
#include <ie_blob.h>
#include <log/log.h>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

bool IENetwork::loadNetwork() {
    ALOGD("%s", __func__);

#if __ANDROID__
    InferenceEngine::Core ie(std::string("/vendor/etc/openvino/plugins.xml"));
#else
    InferenceEngine::Core ie(std::string("/usr/local/lib64/plugins.xml"));
#endif
    std::map<std::string, std::string> config;

    if (mNetwork) {
        mExecutableNw = ie.LoadNetwork(*mNetwork, "CPU");
        ALOGD("LoadNetwork is done....");
        mInferRequest = mExecutableNw.CreateInferRequest();
        ALOGD("CreateInfereRequest is done....");

        mInputInfo = mNetwork->getInputsInfo();
        mOutputInfo = mNetwork->getOutputsInfo();

        //#ifdef NN_DEBUG
        for (auto input : mInputInfo) {
            auto dims = input.second->getTensorDesc().getDims();
            for (auto i : dims) {
                ALOGI(" Dimes : %d", i);
            }
            ALOGI("Name: %s ", input.first.c_str());
        }
        for (auto output : mOutputInfo) {
            auto dims = output.second->getTensorDesc().getDims();
            for (auto i : dims) {
                ALOGI(" Dimes : %d", i);
            }
            ALOGI("Name: %s ", output.first.c_str());
        }
        //#endif
    } else {
        ALOGE("Invalid Network pointer");
        return false;
    }

    return true;
}

// Need to be called before loadnetwork.. But not sure whether need to be called for
// all the inputs in case multiple input / output
void IENetwork::prepareInput(InferenceEngine::Precision precision, InferenceEngine::Layout layout) {
    ALOGE("%s", __func__);

    auto inputInfoItem = *mInputInfo.begin();
    inputInfoItem.second->setPrecision(precision);
    inputInfoItem.second->setLayout(layout);
}

void IENetwork::prepareOutput(InferenceEngine::Precision precision,
                              InferenceEngine::Layout layout) {
    InferenceEngine::DataPtr& output = mOutputInfo.begin()->second;
    output->setPrecision(precision);
    output->setLayout(layout);
}

void IENetwork::setBlob(const std::string& inName, const InferenceEngine::Blob::Ptr& inputBlob) {
    ALOGI("setBlob input or output blob name : %s", inName.c_str());
    mInferRequest.SetBlob(inName, inputBlob);
}

InferenceEngine::TBlob<float>::Ptr IENetwork::getBlob(const std::string& outName) {
    InferenceEngine::Blob::Ptr outputBlob;
    outputBlob = mInferRequest.GetBlob(outName);
    return android::hardware::neuralnetworks::nnhal::As<InferenceEngine::TBlob<float>>(outputBlob);
}

void IENetwork::infer() {
    ALOGI("Infer Network\n");
    mInferRequest.StartAsync();
    mInferRequest.Wait(10000);
    ALOGI("infer request completed");
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
