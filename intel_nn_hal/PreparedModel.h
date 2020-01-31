/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ANDROID_ML_NN_PREPAREDMODEL_H
#define ANDROID_ML_NN_PREPAREDMODEL_H

#include <android/hardware/neuralnetworks/1.2/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.2/types.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hardware/hardware.h>
#include <hidlmemory/mapping.h>
#include <sys/mman.h>
#include <fstream>
#include <string>

#include "Driver.h"
#include "IENetwork.h"

#define EXPL_PAD 1
#define IMPL_PAD 2

using ::android::hardware::MQDescriptorSync;
using ::android::hidl::memory::V1_0::IMemory;
using namespace InferenceEngine;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace {

using time_point = std::chrono::steady_clock::time_point;

auto now() { return std::chrono::steady_clock::now(); };

auto microsecondsDuration(decltype(now()) end, decltype(now()) start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
};

}  // namespace
template <class T>
using vec = std::vector<T>;
typedef uint8_t* memory;

// The type and dimensions of an operand.
struct Shape {
    OperandType type;
    std::vector<uint32_t> dimensions;
    float scale;
    int32_t offset;
};

// Information we maintain about each operand during execution that
// may change during execution.
struct RunTimeOperandInfo {
    // std::string name;
    // uint32_t opIdx;
    // void * opIdx;

    // TODO Storing the type here is redundant, as it won't change during execution.
    OperandType type;
    // The type and dimensions of the operand.  The dimensions can
    // change at runtime.  We include the type because it's useful
    // to pass together with the dimension to the functions implementing
    // the operators.
    std::vector<uint32_t> dimensions;

    float scale;
    int32_t zeroPoint;
    // Where the operand's data is stored.  Check the corresponding
    // location information in the model to figure out if this points
    // to memory we have allocated for an temporary operand.
    uint8_t* buffer;
    // The length of the buffer.
    uint32_t length;
    // Whether this is a temporary variable, a model input, a constant, etc.
    OperandLifeTime lifetime;
    // Keeps track of how many operations have yet to make use
    // of this temporary variable.  When the count is decremented to 0,
    // we free the buffer.  For non-temporary variables, this count is
    // always 0.
    uint32_t numberOfUsesLeft;

    Shape shape() const {
        return Shape{.type = type, .dimensions = dimensions, .scale = scale, .offset = zeroPoint};
    }
};

// Used to keep a pointer to each of the memory pools.
struct RunTimePoolInfo {
    sp<IMemory> memory;
    hidl_memory hidlMemory;
    uint8_t* buffer;

    bool set(const hidl_memory& hidlMemory);
    bool update();
};

bool setRunTimePoolInfosFromHidlMemories(std::vector<RunTimePoolInfo>* poolInfos,
                                         const hidl_vec<hidl_memory>& pools);

// Base class used to create vpu drivers for the NN HAL.  This class
// provides some implementation of the more common functions.
//
// Since these drivers simulate hardware, they must run the computations
// on the CPU.  An actual driver would not do that.
template <typename T_IExecutionCallback>
;
class PreparedModel : public V1_2::IPreparedModel {
public:
    PreparedModel(const Model& model)
        : mTargetDevice(TargetDevice::eMYRIAD),
          mModel(model),
          mNet("nnNet"),
          enginePtr(nullptr),
          mPadreq(EXPL_PAD) {
        g_layer_precision = InferenceEngine::Precision::FP16;
    }

    PreparedModel(const TargetDevice device, const Model& model)
        : mTargetDevice(device),
          mModel(model),
          mNet("nnNet"),
          enginePtr(nullptr),
          mPadreq(EXPL_PAD) {
        if (mTargetDevice == TargetDevice::eCPU || mTargetDevice == TargetDevice::eGPU)
            g_layer_precision = InferenceEngine::Precision::FP32;
        else if (mTargetDevice == TargetDevice::eMYRIAD)
            g_layer_precision = InferenceEngine::Precision::FP16;
        else if (mTargetDevice == TargetDevice::eGNA)
            g_layer_precision = InferenceEngine::Precision::FP32;
        else
            g_layer_precision = InferenceEngine::Precision::UNSPECIFIED;
    }

    ~PreparedModel() override { deinitialize(); }
    bool initialize();
    Return<ErrorStatus> execute(const Request& request,
                                const sp<V1_0::IExecutionCallback>& callback) override;
    Return<ErrorStatus> execute_1_2(const Request& request, MeasureTiming measure,
                                    const sp<V1_2::IExecutionCallback>& callback) override;
    Return<void> executeSynchronously(const Request& request, MeasureTiming measure,
                                      executeSynchronously_cb cb) override;
    Return<void> configureExecutionBurst(
        const sp<V1_2::IBurstCallback>& callback,
        const MQDescriptorSync<V1_2::FmqRequestDatum>& requestChannel,
        const MQDescriptorSync<V1_2::FmqResultDatum>& resultChannel,
        configureExecutionBurst_cb cb) override;

    // Return<ErrorStatus> executeBase(const Request& request, MeasureTiming measure,
    //                             const sp<T_IExecutionCallback>& callback);
    static bool isOperationSupported(const Operation& operation, const Model& model);

protected:
    void deinitialize();
    bool initializeRunTimeOperandInfo();
    Return<ErrorStatus> executeBase(const Request& request, MeasureTiming measure,
                                    const sp<V1_0::IExecutionCallback>& callback);
    Return<ErrorStatus> executeBase_1_2(const Request& request, MeasureTiming measure,
                                        const sp<V1_2::IExecutionCallback>& callback);
    void asyncExecute(const Request& request, MeasureTiming measure, time_point driverStart,
                      const sp<V1_0::IExecutionCallback>& callback);
    void asyncExecute_1_2(const Request& request, MeasureTiming measure, time_point driverStart,
                          const sp<V1_2::IExecutionCallback>& callback);

    bool operationAdd(const Operation& operation);
    bool operationAveragePool2D(const Operation& operation);
    bool operationConCat(const Operation& operation);
    bool operationConv2D(const Operation& operation);
    bool operationDepthwiseConv2D(const Operation& operation);
    bool operationFullyConnected(const Operation& operation);
    bool operationL2Normalization(const Operation& operation);
    bool operationLRN(const Operation& operation);
    bool operationMaxPool2D(const Operation& operation);
    bool operationLogisticSigmoid(const Operation& operation);
    // bool operationLSTM(const Operation& operation);
    bool operationMUL(const Operation& operation);
    bool operationRELU(const Operation& operation);
    bool operationRELU1(const Operation& operation);
    bool operationRELU6(const Operation& operation);
    bool operationReshape(const Operation& operation);
    bool operationSoftmax(const Operation& operation);
    bool operationTANH(const Operation& operation);

    void initializeInput();
    bool finalizeOutput(/*RunTimeOperandInfo* output*/);

    OutputPort handleFusion(const OutputPort& out, int32_t fusedOp);
    template <typename T>
    T GetConstFromBuffer(const uint8_t* buf, uint32_t len);
    template <typename T>
    std::vector<T> GetConstVecFromBuffer(const uint8_t* buf, uint32_t len);
    const uint8_t* GetOperandMemory(const Model& model, uint32_t index, uint32_t& len_out);
    template <typename T>
    T ParseOperationInput(const Model& model, const Operation& operation, uint32_t index);
    template <typename T>
    T GetConstOperand(const Model& model, uint32_t index);
    template <typename T>
    std::vector<T> GetConstVecOperand(const Model& model, uint32_t index);
    virtual Blob::Ptr GetConstOperandAsTensor(int operand_index, int operation_idx);
    virtual Blob::Ptr GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t* buf,
                                            uint32_t& len);
    virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index);
    void SetOperandMemory(const Model& model, uint32_t index, uint32_t& len_out,
                          const uint8_t* buf);
    void SetOperandFromTensor(uint8_t* buf, uint32_t& length, Blob::Ptr infOutput);
    bool isConst(int index);
    OutputPort getPort(int index);

    TargetDevice mTargetDevice;
    Model mModel;
    std::vector<RunTimeOperandInfo> mOperands;
    std::vector<RunTimePoolInfo> mPoolInfos;
    IRDocument mNet;
    std::vector<OutputPort> mPorts;  // typedef std::shared_ptr<Data> DataPtr;
    ExecuteNetwork* enginePtr;
    uint32_t mPadreq;
};

class VpuPreparedModel : public PreparedModel {
public:
    VpuPreparedModel(const Model& model) : PreparedModel(TargetDevice::eMYRIAD, model) {}

    virtual Blob::Ptr GetConstOperandAsTensor(int operand_index, int operation_idx) override;
    virtual Blob::Ptr GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t* buf,
                                            uint32_t& len) override;
    virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index) override;
};

class CpuPreparedModel : public PreparedModel {
public:
    CpuPreparedModel(const Model& model) : PreparedModel(TargetDevice::eCPU, model) {}

    virtual Blob::Ptr GetConstOperandAsTensor(int operand_index, int operation_idx) override;
    virtual Blob::Ptr GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t* buf,
                                            uint32_t& len) override;
    virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index) override;
};

class GpuPreparedModel : public PreparedModel {
public:
    GpuPreparedModel(const Model& model) : PreparedModel(TargetDevice::eGPU, model) {}

    virtual Blob::Ptr GetConstOperandAsTensor(int operand_index, int operation_idx) override;
    virtual Blob::Ptr GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t* buf,
                                            uint32_t& len) override;
    virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index) override;
};

class GnaPreparedModel : public PreparedModel {
public:
    GnaPreparedModel(const Model& model)
          :PreparedModel(TargetDevice::eGNA, model) {
    }

    virtual Blob::Ptr GetConstOperandAsTensor(int operand_index, int operation_idx) override;
    virtual Blob::Ptr GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len) override;
    virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index) override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // ANDROID_ML_NN_PREPAREDMODEL_H
