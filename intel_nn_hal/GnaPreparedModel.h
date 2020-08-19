#ifndef ANDROID_ML_NN_GNA_PREPAREDMODEL_H
#define ANDROID_ML_NN_GNA_PREPAREDMODEL_H

#include <android/hardware/neuralnetworks/1.2/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.2/types.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hardware/hardware.h>
#include <hidlmemory/mapping.h>
#include <sys/mman.h>
#include <fstream>
#include <string>

#include "GnaNetwork.h"
#include "PreparedModel.h"

#define EXPL_PAD 1
#define IMPL_PAD 2

using ::android::hardware::MQDescriptorSync;
using ::android::hidl::memory::V1_0::IMemory;
using namespace InferenceEngine;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class GnaPreparedModel : public PreparedModel {

    struct LayerInfo {
        std::string layerName;
        bool memoryLayer;

        LayerInfo(std::string layer, bool memory):layerName(layer),memoryLayer(memory){}
    };

    std::map<uint32_t, LayerInfo> mInputPorts;
    std::map<uint32_t, std::string> mOutputToLayerMap;
    std::vector<uint32_t> mlayerInputIndices; /* to be filled during the infer call. Need to optimize */

    IRBuilder::ModelBuilder* mBuilderModel;
    GnaNetwork* gnaPluginPtr;

public:
    GnaPreparedModel(const Model& model) : PreparedModel("GNA", model), gnaPluginPtr(nullptr) {}
    ~GnaPreparedModel()  {  deinitialize(); }

    virtual bool initialize() override;
    virtual bool operationFullyConnected(const Operation& operation) override;

    virtual Blob::Ptr GetConstOperandAsTensor(int operand_index, int operation_idx) override;
    virtual Blob::Ptr GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len) override;
    virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index) override;

    virtual Return<ErrorStatus> execute(const Request& request,
                                const sp<V1_0::IExecutionCallback>& callback) override {
        return executeBase(request, MeasureTiming::NO, callback);
    }

    virtual void initializeInput() override;
    virtual bool finalizeOutput() override;

    bool operationLSTM(const Operation& operation);

protected:
    void deinitialize();
    virtual Return<ErrorStatus> executeBase(const Request& request, MeasureTiming measure,
                                    const sp<V1_0::IExecutionCallback>& callback) override;

    void asyncExecute(const Request& request, MeasureTiming measure, time_point driverStart,
                      const sp<V1_0::IExecutionCallback>& callback);

};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // ANDROID_ML_NN_PREPAREDMODEL_H
