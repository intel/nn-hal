#ifndef __DEVICE_PLUGIN_H
#define __DEVICE_PLUGIN_H

#include <ie_cnn_network.h>
#include <ie_core.hpp>
#include <ie_executable_network.hpp>
#include <ie_infer_request.hpp>
#include <ie_input_info.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/infer_request.hpp>
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
    virtual ov::InferRequest getInferRequest() = 0;
    virtual void infer() = 0;
    virtual void queryState() = 0;
    virtual ov::Tensor getBlob(const std::string& outName) = 0;
    virtual ov::Tensor getInputBlob(const std::size_t index) = 0;
    virtual ov::Tensor getOutputBlob(const std::size_t index) = 0;
};

// Abstract this class for all accelerators
class IENetwork : public IIENetwork {
private:
    IntelDeviceType mTargetDevice;
    std::shared_ptr<ov::Model> mNetwork;
    ov::CompiledModel compiled_model;
    ov::InferRequest mInferRequest;

public:
    IENetwork(IntelDeviceType device, std::shared_ptr<ov::Model> network)
        : mTargetDevice(device), mNetwork(network) {}

    virtual bool loadNetwork();
    ov::Tensor getBlob(const std::string& outName);
    ov::Tensor getInputBlob(const std::size_t index);
    ov::Tensor getOutputBlob(const std::size_t index);
    ov::InferRequest getInferRequest() { return mInferRequest; }
    void queryState() {}
    void infer();
};

}  // namespace android::hardware::neuralnetworks::nnhal

#endif