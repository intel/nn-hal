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

#ifndef ANDROID_ML_NN_EXECUTOR_H
#define ANDROID_ML_NN_EXECUTOR_H

#include <android/hardware/neuralnetworks/1.0/IPreparedModel.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hidlmemory/mapping.h>
#include <hardware/hardware.h>
#include <sys/mman.h>
#include <string>
#include <fstream>

#include "IENetwork.h"

using ::android::hidl::memory::V1_0::IMemory;
using namespace IRBuilder;
using namespace InferenceEngine;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace driver {
namespace executor {


template <class T> using  vec = std::vector<T>;
typedef uint8_t * memory;

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
    //std::string name;
    //uint32_t opIdx;
    //void * opIdx;

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



// This class is used to execute a model
class Executor {
public:
    Executor()
          :mTargetDevice(TargetDevice::eMYRIAD), mNet("nnNet"), enginePtr(nullptr) {
        IRBuilder::g_layer_precision = InferenceEngine::Precision::FP16;
    }

    Executor(const TargetDevice device)
          :mTargetDevice(device), mNet("nnNet"), enginePtr(nullptr) {
        if (mTargetDevice == TargetDevice::eCPU)
           IRBuilder::g_layer_precision = InferenceEngine::Precision::FP32;
        else if (mTargetDevice == TargetDevice::eMYRIAD)
           IRBuilder::g_layer_precision = InferenceEngine::Precision::FP16;
        else
           IRBuilder::g_layer_precision = InferenceEngine::Precision::UNSPECIFIED;
    }

    ~Executor() {deinitialize();}
    //bool initialize();
    // Executes the model. The results will be stored at the locations
    // specified in the constructor.
    // The model must outlive the executor.  We prevent it from being modified
    // while this is executing.
    int run(const Model& model, const Request& request,
            std::vector<RunTimePoolInfo>& modelPoolInfos,
            std::vector<RunTimePoolInfo>& requestPoolInfos);

protected:
    void deinitialize();
    bool initializeRunTimeInfo(const std::vector<RunTimePoolInfo>& modelPoolInfos,
                                            const std::vector<RunTimePoolInfo>& requestPoolInfos);

    bool executeOperation(const Operation& operation);

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
    //bool operationLSTM(const Operation& operation);
    bool operationMUL(const Operation& operation);
    bool operationRELU(const Operation& operation);
    bool operationRELU1(const Operation& operation);
    bool operationRELU6(const Operation& operation);
    bool operationReshape(const Operation& operation);
    bool operationSoftmax(const Operation& operation);
    bool operationTANH(const Operation& operation);

    void initializeInput();
    void finalizeOutput(/*RunTimeOperandInfo* output*/);

    OutputPort handleFusion(const OutputPort &out, int32_t fusedOp);
    template<typename T>
    T GetConstFromBuffer(const uint8_t *buf, uint32_t len);
    template<typename T>
    std::vector<T> GetConstVecFromBuffer(const uint8_t *buf, uint32_t len);
    const uint8_t *GetOperandMemory(const Model *model, uint32_t index, uint32_t &len_out);
    template <typename T>
    T ParseOperationInput(const Model *model, const Operation& operation, uint32_t index);
    template <typename T>
    T GetConstOperand(const Model *model, uint32_t index);
    template <typename T>
    std::vector<T> GetConstVecOperand(const Model *model, uint32_t index);
    virtual Blob::Ptr GetConstOperandAsTensor(uint32_t index);
    virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index);
    virtual Blob::Ptr GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len);
    void SetOperandMemory(const Model *model, uint32_t index, uint32_t &len_out, const uint8_t *buf);
    void SetOperandFromTensor(uint8_t* buf, uint32_t &length, Blob::Ptr infOutput);
    bool isConst(int index);
    OutputPort getPort(int index);

    TargetDevice mTargetDevice;
    std::vector<RunTimeOperandInfo> mOperands;
    IRDocument mNet;
    std::vector<OutputPort> mPorts;  //typedef std::shared_ptr<Data> DataPtr;
    ExecuteNetwork* enginePtr;

    // The model and the request that we'll execute. Only valid while run()
    // is being executed.
    const Model* mModel = nullptr;
    const Request* mRequest = nullptr;


};

class VpuExecutor : public Executor {
public:
    VpuExecutor()
          :Executor(TargetDevice::eMYRIAD) {
    }

    virtual Blob::Ptr GetConstOperandAsTensor(uint32_t index) override;
    virtual Blob::Ptr GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len) override;
    virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index) override;
};

class CpuExecutor : public Executor {
public:
    CpuExecutor()
          :Executor(TargetDevice::eCPU) {
    }

    virtual Blob::Ptr GetConstOperandAsTensor(uint32_t index) override;
    virtual Blob::Ptr GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len) override;
    virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index) override;
};



class PreparedModel : public IPreparedModel {
public:
    PreparedModel(const Model& model) : mModel(model), mTargetDevice(TargetDevice::eMYRIAD) {
        IRBuilder::g_layer_precision = InferenceEngine::Precision::FP16;
    }
    PreparedModel(const TargetDevice device, const Model& model)
          :mTargetDevice(device), mModel(model) {
        if (mTargetDevice == TargetDevice::eCPU)
           IRBuilder::g_layer_precision = InferenceEngine::Precision::FP32;
        else if (mTargetDevice == TargetDevice::eMYRIAD)
           IRBuilder::g_layer_precision = InferenceEngine::Precision::FP16;
        else
           IRBuilder::g_layer_precision = InferenceEngine::Precision::UNSPECIFIED;
    }
    ~PreparedModel() override {}
    bool initialize();
    static bool isOperationSupported(const Operation& operation, const Model& model);
    Return<ErrorStatus> execute(const Request& request,
                                const sp<IExecutionCallback>& callback) override;

private:
    void asyncExecute(const Request& request, const sp<IExecutionCallback>& callback);

    Model mModel;
    std::vector<RunTimePoolInfo> mPoolInfos;
    TargetDevice mTargetDevice;

};

class VpuPreparedModel : public PreparedModel {
public:
    VpuPreparedModel(const Model& model)
          :PreparedModel(TargetDevice::eMYRIAD, model) {
    }

};

class CpuPreparedModel : public PreparedModel {
public:
    CpuPreparedModel(const Model& model)
          :PreparedModel(TargetDevice::eCPU, model) {
    }

};

}
}  // namespace driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif // ANDROID_ML_NN_PREPAREDMODEL_H
