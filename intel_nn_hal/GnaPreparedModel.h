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
        DeviceType execDevice;

        LayerInfo(std::string layer, bool memory):layerName(layer),memoryLayer(memory), execDevice(DeviceType::GNA){}
        LayerInfo(std::string layer, bool memory, DeviceType device):layerName(layer),memoryLayer(memory), execDevice(device){}
    };

    struct HalLayerInfo {
        std::string inLayerName;
        std::string outLayerName;
        bool memoryLayer;
        DeviceType inDevice;
        DeviceType outDevice;

        HalLayerInfo(std::string inLayer, DeviceType inDev,
                     std::string outLayer, DeviceType outDev,
                     bool memory):inLayerName(inLayer), inDevice(inDev),
                                  outLayerName(outLayer), outDevice(outDev),
                                  memoryLayer(memory){}

        void setInputNode(std::string inLayer, DeviceType dev) {
            inLayerName = inLayer;
            inDevice = dev;
        }

        void setOutputNode(std::string outLayer, DeviceType dev) {
            outLayerName = outLayer;
            inDevice = dev;
        }
    };

    std::map<uint32_t, LayerInfo> mInputPorts;
    std::map<uint32_t, LayerInfo> mOutputToLayerMap;
    std::map<uint32_t, HalLayerInfo> mIntermediateLayerMap;
    std::vector<uint32_t> mlayerInputIndices; /* to be filled during the infer call. Need to optimize */
    //std::vector<uint32_t> mlayerIntermediateIndices;
    std::vector<uint32_t> mModelInputIndices;
    std::vector<uint32_t> mModelOutputIndices;

    IRBuilder::ModelBuilder* mBuilderModel;
    GnaNetwork* gnaPluginPtr;
    std::vector<IRBlob::Ptr> mModelIRBlobs; /*intermediate IRBlobs to be deallocated after network is loaded into the Plugin */

    std::vector<OpContainer*> mNwManager;

    bool isDecoderNw = false;
    bool isEnc0Nw = false;
    bool isEnc1Nw = false;
    bool isJointNw = false;

    std::string modelNameStr;
#ifdef PERF_COUNTERS
    metrics runtimeMetrics;
#endif
public:
#ifdef CACHING
    GnaPreparedModel(const Model& model) : PreparedModel("GNA", model), gnaPluginPtr(nullptr), mBuilderModel(nullptr),
                                            isDecoderNw(false), isEnc0Nw(false), isEnc1Nw(false), isJointNw(false) {
#ifdef PERF_COUNTERS
        runtimeMetrics.reset();
#endif
    }
	GnaPreparedModel() : PreparedModel("GNA"), gnaPluginPtr(nullptr), mBuilderModel(nullptr),
                         isDecoderNw(false), isEnc0Nw(false), isEnc1Nw(false), isJointNw(false) {
#ifdef PERF_COUNTERS
        runtimeMetrics.reset();
#endif
    }

    virtual bool initializeFromCache(const hidl_vec<hidl_handle>& modelCache, const HidlToken& token) override;
#else
    GnaPreparedModel(const Model& model) : PreparedModel("GNA", model), gnaPluginPtr(nullptr), mBuilderModel(nullptr),
                                            isDecoderNw(false), isEnc0Nw(false), isEnc1Nw(false) {
#ifdef PERF_COUNTERS
        runtimeMetrics.reset();
#endif
    }
#endif

    ~GnaPreparedModel()  {
#ifdef PERF_COUNTERS
        std::cout << " ********* " << modelNameStr
                    << " ***********" << std::endl;
        runtimeMetrics.print();
#endif
        deinitialize();
    }

    virtual bool initialize(const hidl_vec<hidl_handle>& modelCache, const HidlToken& token) override;
    virtual bool operationFullyConnected(const Operation& operation) override;
    virtual bool operationAdd(const Operation& operation) override;
    virtual bool operationTANH(const Operation& operation) override;

    virtual Blob::Ptr GetConstOperandAsTensor(int operand_index, int operation_idx) override;
    virtual Blob::Ptr GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len) override;
    virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index) override;

    virtual Return<V1_0_ErrorStatus> execute(const V1_0_Request& request,
                                const sp<V1_0::IExecutionCallback>& callback) override {
        return V1_0::ErrorStatus::NONE;
    }

    Return<V1_3::ErrorStatus> execute_1_3(const V1_3::Request& request,
                                          V1_2::MeasureTiming measure,
                                          const V1_3::OptionalTimePoint&,
                                          const V1_3::OptionalTimeoutDuration&,
                                          const sp<V1_3::IExecutionCallback>& callback) override  {
        return executeBase_V1_3(request, MeasureTiming::NO, callback);
    }

    Return<void> executeSynchronously_1_3(const V1_3::Request &request,
                                          V1_2::MeasureTiming measure,
                                          const V1_3::OptionalTimePoint& deadline,
                                          const V1_3::OptionalTimeoutDuration& loopTimeoutDuration,
                                          V1_3::IPreparedModel::executeSynchronously_1_3_cb cb) override {

        Timing kNoTiming_1 = {.timeOnDevice = UINT64_MAX, .timeInDriver = UINT64_MAX};

        auto [status, outputShapes, timing] = executeSynchronouslyBase(request, measure, deadline, loopTimeoutDuration);
        cb(status, std::move(outputShapes), timing);
        return Void();
    };

    virtual void initializeInput() override;

    virtual bool finalizeOutput() override;
    std::tuple<ErrorStatus, hidl_vec<V1_2::OutputShape>, Timing>
    executeSynchronouslyBase(const V1_3::Request& request, V1_2::MeasureTiming measure,
                  const V1_3::OptionalTimePoint& halDeadline,
                  const V1_3::OptionalTimeoutDuration& loopTimeoutDuration);
    std::tuple<ErrorStatus, hidl_vec<V1_2::OutputShape>, Timing>
        syncExecute(const V1_3::Request& request, MeasureTiming measure, time_point driverStart);
    void executeModel(const V1_3::Request& request);

    void initializeInput(std::vector<uint32_t>& indexVec);
    bool operationLSTM(const Operation& operation);
    bool operationQuantizedLSTM(const Operation& operation);
    void setDims(const uint32_t idx, const std::vector<unsigned long> dims);
    BaseOp* operationEmbeddingLookup(const Operation& operation);
    BaseOp* operationDequantize(const Operation& operation, bool dummyOp = false);
    BaseOp* operationQuantize(const Operation& operation, bool dummyOp = false);
    bool isRnnT() {
        return (isDecoderNw ? true : (isEnc0Nw ? true : (isEnc1Nw ? true : (isJointNw ? true : false))));
    }


protected:
    void deinitialize();
    void asyncExecute(const V1_3::Request& request, MeasureTiming measure, time_point driverStart,
                      const sp<V1_3::IExecutionCallback>& callback);

    bool constructGNAGraph(std::pair<int, int> indices);
    OpContainer* constructCpuGraph(std::pair<int, int> indices);
    BaseOp* getCpuOpFromLayerName(std::string layer);
        Return<V1_3::ErrorStatus> executeBase_V1_3(const V1_3::Request& request, MeasureTiming measure,
                                    const sp<V1_3::IExecutionCallback>& callback);
    Return<V1_0::ErrorStatus> executeBase(const V1_0::Request& request, MeasureTiming measure,
                                    const sp<V1_0::IExecutionCallback>& callback) override;

    bool updateMemoryAfterCPUGraphExecution(const V1_0_Request& request);
    bool updateMemoryAfterCPUGraphExecution(const V1_0_Request& request, uint32_t index);
    bool updateMemoryAfterGNAGraphExecution(const V1_0_Request& request);
    bool updateMemoryAfterGraphExecution(const V1_3::Request& request);


    std::vector<RunTimePoolInfo> mRuntimeRequestPoolInfos;

    Blob::Ptr getBlobFromMemoryPool(uint32_t index, const V1_3::Request& request, OperandType& opType);
    RunTimeOperandInfo& getOperandFromMemoryPool(uint32_t index, const V1_3::Request& request);
    void executeGnaGraph();
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // ANDROID_ML_NN_GNA_PREPAREDMODEL_H
