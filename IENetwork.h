#ifndef __DEVICE_PLUGIN_H
#define __DEVICE_PLUGIN_H

#include <ie_cnn_network.h>
#include <ie_core.hpp>
#include <ie_executable_network.hpp>
#include <ie_infer_request.hpp>
#include <ie_input_info.hpp>
#include <vector>

#include "utils.h"
// #include "ie_blob.h"
// #include "ie_common.h"
// #include "ie_core.hpp"
// #include "inference_engine.hpp"
namespace android::hardware::neuralnetworks::nnhal {

class IIENetwork {
public:
    virtual ~IIENetwork() {}
    virtual bool loadNetwork() = 0;
    virtual InferenceEngine::InferRequest getInferRequest() = 0;
    virtual void infer() = 0;
    virtual void queryState() = 0;
    virtual InferenceEngine::TBlob<float>::Ptr getBlob(const std::string& outName) = 0;
    virtual void prepareInput(InferenceEngine::Precision precision,
                              InferenceEngine::Layout layout) = 0;
    virtual void prepareOutput(InferenceEngine::Precision precision,
                               InferenceEngine::Layout layout) = 0;
    virtual void setBlob(const std::string& inName,
                         const InferenceEngine::Blob::Ptr& inputBlob) = 0;
};

// Abstract this class for all accelerators
class IENetwork : public IIENetwork {
private:
    IntelDeviceType mTargetDevice;
    std::shared_ptr<InferenceEngine::CNNNetwork> mNetwork;
    InferenceEngine::ExecutableNetwork mExecutableNw;
    InferenceEngine::InferRequest mInferRequest;
    InferenceEngine::InputsDataMap mInputInfo;
    InferenceEngine::OutputsDataMap mOutputInfo;

public:
    IENetwork(IntelDeviceType device, std::shared_ptr<InferenceEngine::CNNNetwork> network)
        : mTargetDevice(device), mNetwork(network) {}

    virtual bool loadNetwork();
    void prepareInput(InferenceEngine::Precision precision, InferenceEngine::Layout layout);
    void prepareOutput(InferenceEngine::Precision precision, InferenceEngine::Layout layout);
    void setBlob(const std::string& inName, const InferenceEngine::Blob::Ptr& inputBlob);
    InferenceEngine::TBlob<float>::Ptr getBlob(const std::string& outName);
    InferenceEngine::InferRequest getInferRequest() { return mInferRequest; }
    void queryState() {}
    void infer();
};

}  // namespace android::hardware::neuralnetworks::nnhal

#endif
