#ifndef ANDROID_ML_NN_GNA_PREPAREDMODEL_H
#define ANDROID_ML_NN_GNA_PREPAREDMODEL_H

#include <android/hardware/neuralnetworks/1.2/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.2/types.h>
#include <android/hardware/neuralnetworks/1.3/types.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hidlmemory/mapping.h>
#include <sys/mman.h>
#include <fstream>
#include <string>

#include "GnaNetwork.h"
#include "PreparedModel.h"
#include "Utils.h"

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
    std::vector<uint32_t> mModelInputIndices;
    std::vector<uint32_t> mModelOutputIndices;

    IRBuilder::ModelBuilder* mBuilderModel;
    GnaNetwork* gnaPluginPtr;

#ifdef PERF_COUNTERS
    bool isDecoderNw, isEnc0Nw, isEnc1Nw;
    std::string modelNameStr;
    metrics runtimeMetrics;
#endif
public:
#ifdef CACHING
    GnaPreparedModel(const Model& model) : PreparedModel("GNA", model), gnaPluginPtr(nullptr), mBuilderModel(nullptr),
                                            isDecoderNw(false), isEnc0Nw(false), isEnc1Nw(false) {
        runtimeMetrics.reset();
    }
	GnaPreparedModel() : PreparedModel("GNA"), gnaPluginPtr(nullptr), mBuilderModel(nullptr),
                                            isDecoderNw(false), isEnc0Nw(false), isEnc1Nw(false) {
        runtimeMetrics.reset();
    }
    ~GnaPreparedModel()  {
        std::string nw_name = isDecoderNw?"Decoder":isEnc0Nw?"Encoder0":"Encoder1";
        std::cout << " ********* " << nw_name
                    << " ***********" << std::endl;
        runtimeMetrics.print();
        deinitialize();
    }
    virtual bool initializeFromCache(const hidl_vec<hidl_handle>& modelCache, const HidlToken& token) override;
#else
    GnaPreparedModel(const Model& model) : PreparedModel("GNA", model), gnaPluginPtr(nullptr), mBuilderModel(nullptr) {
    }

    ~GnaPreparedModel()  {
        deinitialize();
    }
#endif

    virtual bool initialize(const hidl_vec<hidl_handle>& modelCache, const HidlToken& token) override;
    virtual bool operationFullyConnected(const Operation& operation) override;

    virtual Blob::Ptr GetConstOperandAsTensor(int operand_index, int operation_idx) override;
    virtual Blob::Ptr GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len) override;
    virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index) override;

    virtual Return<V1_0_ErrorStatus> execute(const V1_0_Request& request,
                                const sp<V1_0::IExecutionCallback>& callback) override {
        return executeBase(request, MeasureTiming::NO, callback);
    }

    virtual void initializeInput() override;
    void initializeInput(std::vector<uint32_t>& indexVec);
    virtual bool finalizeOutput() override;

    bool operationLSTM(const Operation& operation);
    bool operationQuantizedLSTM(const Operation& operation);

protected:
    void deinitialize();
    virtual Return<V1_0_ErrorStatus> executeBase(const V1_0_Request& request, MeasureTiming measure,
                                    const sp<V1_0::IExecutionCallback>& callback) override;

    void asyncExecute(const V1_0_Request& request, MeasureTiming measure, time_point driverStart,
                      const sp<V1_0::IExecutionCallback>& callback);

};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // ANDROID_ML_NN_PREPAREDMODEL_H
