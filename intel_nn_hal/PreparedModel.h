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
#include <android/hardware/neuralnetworks/1.3/types.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hidlmemory/mapping.h>
#include <sys/mman.h>
#include <fstream>
#include <string>

#include "Driver.h"
#include "IENetwork.h"
#include "BuilderNetwork.h"
#include "IRBuilder.h"
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

namespace {

using time_point = std::chrono::steady_clock::time_point;

auto now() { return std::chrono::steady_clock::now(); };

auto microsecondsDuration(decltype(now()) end, decltype(now()) start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
};

auto millisecondsDuration(decltype(now()) end, decltype(now()) start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
};
}  // namespace

// Base class used to create vpu drivers for the NN HAL.  This class
// provides some implementation of the more common functions.
//
// Since these drivers simulate hardware, they must run the computations
// on the CPU.  An actual driver would not do that.
class PreparedModel : public V1_0::IPreparedModel {
public:
    PreparedModel(const Model& model)
        : mTargetDevice("MYRIAD"),
          mModel(model),
          mNet("nnNet"),
          enginePtr(nullptr),
          mPadreq(EXPL_PAD) {
        g_layer_precision = InferenceEngine::Precision::FP16;
    }

    PreparedModel(const std::string device, const Model& model)
        : mTargetDevice(device),
          mModel(model),
          mNet("nnNet"),
          enginePtr(nullptr),
          mPadreq(EXPL_PAD) {
        if (mTargetDevice == "CPU" || mTargetDevice == "GPU")
            g_layer_precision = InferenceEngine::Precision::FP32;
        else if (mTargetDevice == "MYRIAD")
            g_layer_precision = InferenceEngine::Precision::FP16;
        else if (mTargetDevice == "GNA")
            g_layer_precision = InferenceEngine::Precision::FP32;
        else
            g_layer_precision = InferenceEngine::Precision::UNSPECIFIED;
    }

    ~PreparedModel() override { deinitialize(); }
    virtual bool initialize();
    virtual Return<V1_0_ErrorStatus> execute(const V1_0_Request& request,
                                const sp<V1_0::IExecutionCallback>& callback) override;

    // Return<ErrorStatus> executeBase(const Request& request, MeasureTiming measure,
    //                             const sp<T_IExecutionCallback>& callback);
    static bool isOperationSupported(const Operation& operation, const Model& model, const std::string& device);

protected:
    void deinitialize();
    bool initializeRunTimeOperandInfo();
    virtual Return<V1_0_ErrorStatus> executeBase(const V1_0_Request& request, MeasureTiming measure,
                                    const sp<V1_0::IExecutionCallback>& callback);

    void asyncExecute(const V1_0_Request& request, MeasureTiming measure, time_point driverStart,
                      const sp<V1_0::IExecutionCallback>& callback);

    bool operationAdd(const Operation& operation);
    bool operationAveragePool2D(const Operation& operation);
    bool operationConCat(const Operation& operation);
    bool operationConv2D(const Operation& operation);
    bool operationDepthwiseConv2D(const Operation& operation);
    virtual bool operationFullyConnected(const Operation& operation);
    bool operationL2Normalization(const Operation& operation);
    bool operationLRN(const Operation& operation);
    bool operationMaxPool2D(const Operation& operation);
    bool operationLogisticSigmoid(const Operation& operation);
    bool operationMUL(const Operation& operation);
    bool operationRELU(const Operation& operation);
    bool operationRELU1(const Operation& operation);
    bool operationRELU6(const Operation& operation);
    bool operationReshape(const Operation& operation);
    bool operationSoftmax(const Operation& operation);
    bool operationTANH(const Operation& operation);

    virtual void initializeInput();
    virtual bool finalizeOutput(/*RunTimeOperandInfo* output*/);

    OutputPort handleFusion(const OutputPort& out, int32_t fusedOp);
    template <typename T>
    T GetConstFromBuffer(const uint8_t* buf, uint32_t len) {
    VLOG(L1, "buf: %p, len: %d", buf, len);
    if (len != sizeof(T)) {
        ALOGE("fix me: typeid(T).name() is %d should be %d bytes", len, sizeof(T));
        // fix me if buffer is of type float and if float and V1_0_OperandLifeTime::CONSTANT_REFERENCE
        nnAssert(false);
    }
    return *(T*)(buf);
    }
    template <typename T>
    std::vector<T> GetConstVecFromBuffer(const uint8_t* buf, uint32_t len);
    const uint8_t* GetOperandMemory(const Model& model, uint32_t index, uint32_t& len_out);
    //template <typename T>
    //T ParseOperationInput(const Model& model, const Operation& operation, uint32_t index);
    template <typename T>
	T ParseOperationInput(const Model& model, const Operation& operation,
					     uint32_t index) {
	    uint32_t inputIndex = operation.inputs[index];
	    const auto operand = mModel.main.operands[inputIndex];
	    VLOG("operand index = %d", inputIndex);
	    const auto value = GetConstOperand<T>(model, inputIndex);
	    VLOG(L1, "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
	    VLOG(L1, "Operation input index: %d, operand index: %d", index, inputIndex);
	    VLOG(L1, "Operation: %s", toString(operation).c_str());
	    //printHelper<T>::print(value, toString(operand).c_str());
	    VLOG(L1, "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
	    return value;
	}
    int64_t ParseOperationInput_i8(const Model& model, const Operation& operation, uint32_t index) {
	return ParseOperationInput<int8_t>(model, operation, index);
    }
    template <typename T>
    T GetConstOperand(const Model& model, uint32_t index) {
        uint32_t len;
        const uint8_t* buf = GetOperandMemory(model, index, len);
        return GetConstFromBuffer<T>(buf, len);
    }
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
    bool isOperandDataNull(int index);

    //TargetDevice mTargetDevice;
    std::string mTargetDevice;
    Model mModel;
    std::vector<RunTimeOperandInfo> mOperands;
    std::vector<RunTimePoolInfo> mPoolInfos;
    IRDocument mNet;
    std::vector<OutputPort> mPorts;  // typedef std::shared_ptr<Data> DataPtr;
    ExecuteNetwork* enginePtr;
    uint32_t mPadreq;

    InferenceEngine::ICNNNetwork  *mCnnNetbuilder;
    std::map<int, IRBlob::Ptr> mOpIndex2BlobMap;
    std::unordered_map<std::string, IRBlob::Ptr> mLayerNameBlobMap;
};

class VpuPreparedModel : public PreparedModel {
public:
    VpuPreparedModel(const Model& model) : PreparedModel("MYRIAD", model) {}

    virtual Blob::Ptr GetConstOperandAsTensor(int operand_index, int operation_idx) override;
    virtual Blob::Ptr GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t* buf,
                                            uint32_t& len) override;
    virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index) override;
};

class CpuPreparedModel : public PreparedModel {
public:
    CpuPreparedModel(const Model& model) : PreparedModel("CPU", model) {}

    virtual Blob::Ptr GetConstOperandAsTensor(int operand_index, int operation_idx) override;
    virtual Blob::Ptr GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t* buf,
                                            uint32_t& len) override;
    virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index) override;
};

class GpuPreparedModel : public PreparedModel {
public:
    GpuPreparedModel(const Model& model) : PreparedModel("GPU", model) {}

    virtual Blob::Ptr GetConstOperandAsTensor(int operand_index, int operation_idx) override;
    virtual Blob::Ptr GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t* buf,
                                            uint32_t& len) override;
    virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index) override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // ANDROID_ML_NN_PREPAREDMODEL_H
