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
    IRBuilder::ModelBuilder* mBuilderModel;
    GnaNetwork* gnaPluginPtr;

public:
    GnaPreparedModel(const Model& model) : PreparedModel("GNA", model) {}
    ~GnaPreparedModel() override {  deinitialize(); }

    virtual bool initialize() override;
    virtual bool operationFullyConnected(const Operation& operation) override;

    virtual Blob::Ptr GetConstOperandAsTensor(int operand_index, int operation_idx) override;
    virtual Blob::Ptr GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len) override;
    virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index) override;

    virtual Return<ErrorStatus> execute(const Request& request,
                                const sp<V1_0::IExecutionCallback>& callback) override {
        return executeBase(request, MeasureTiming::NO, callback);
    }

    virtual Return<ErrorStatus> execute_1_2(const Request& request, MeasureTiming measure,
                                    const sp<V1_2::IExecutionCallback>& callback) override {
        return executeBase_1_2(request, measure, callback);
    }

    virtual Return<void> executeSynchronously(const Request& request, MeasureTiming measure,
                                                 executeSynchronously_cb cb) override;

    virtual void initializeInput() override;
    virtual bool finalizeOutput() override;

    bool operationLSTM(const Operation& operation);

protected:
    void deinitialize();
    virtual Return<ErrorStatus> executeBase(const Request& request, MeasureTiming measure,
                                    const sp<V1_0::IExecutionCallback>& callback) override;
    virtual Return<ErrorStatus> executeBase_1_2(const Request& request, MeasureTiming measure,
                                        const sp<V1_2::IExecutionCallback>& callback) override;
    void asyncExecute_lstm(const Request& request, MeasureTiming measure, time_point driverStart,
                          const sp<V1_2::IExecutionCallback>& callback);
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // ANDROID_ML_NN_PREPAREDMODEL_H