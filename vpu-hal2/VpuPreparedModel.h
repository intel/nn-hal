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

#ifndef ANDROID_ML_NN_VPU_PREPAREDMODEL_H
#define ANDROID_ML_NN_VPU_PREPAREDMODEL_H

//#include "halinterfaces.h"
/*
#include <android/hardware/neuralnetworks/1.0/IDevice.h>
#include <android/hardware/neuralnetworks/1.0/IExecutionCallback.h>
#include <android/hardware/neuralnetworks/1.0/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.0/IPreparedModelCallback.h>
#include <android/hardware/neuralnetworks/1.0/types.h>
#include <android/hidl/allocator/1.0/IAllocator.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hidlmemory/mapping.h>
*/

#include <android/hardware/neuralnetworks/1.0/IPreparedModel.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hidlmemory/mapping.h>
#include <hardware/hardware.h>
#include <sys/mman.h>
#include <string>

//#include <mvnc.h>

//vpu include
#include "vpu_plugin.hpp"
#include <fstream>

using ::android::hidl::memory::V1_0::IMemory;
using namespace IRBuilder;
using namespace InferenceEngine;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace vpu_driver {

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



// Base class used to create vpu drivers for the NN HAL.  This class
// provides some implementation of the more common functions.
//
// Since these drivers simulate hardware, they must run the computations
// on the CPU.  An actual driver would not do that.
class VpuPreparedModel : public IPreparedModel {
public:
    VpuPreparedModel(const Model& model)
          : // Make a copy of the model, as we need to preserve it.
            mModel(model), mNet("nnNet") {
	}
    ~VpuPreparedModel() override {deinitialize();}
    bool initialize();
    Return<ErrorStatus> execute(const Request& request,
                                const sp<IExecutionCallback>& callback) override;
    static bool isOperationSupported(const Operation& operation, const Model& model);
    static bool validModel(const Model& model);
    static bool validateRequest(const Request& request, const Model& model);

private:
    void deinitialize();
    bool initializeRunTimeOperandInfo();
    void asyncExecute(const Request& request, const sp<IExecutionCallback>& callback);
    void convertModel(IRDocument &mNet);

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

    void initializeInput(RunTimeOperandInfo* input);
    void finalizeOutput(/*RunTimeOperandInfo* output*/);

    OutputPort handleFusion(const OutputPort &out, int32_t fusedOp);
    template<typename T>
    T GetConstFromBuffer(const uint8_t *buf, uint32_t len);
    template<typename T>
    std::vector<T> GetConstVecFromBuffer(const uint8_t *buf, uint32_t len);
    const uint8_t *GetOperandMemory(const Model &model, uint32_t index, uint32_t &len_out);
    template <typename T>
    T GetConstOperand(const Model &model, uint32_t index);
    template <typename T>
    std::vector<T> GetConstVecOperand(const Model &model, uint32_t index);
    IRBlob::Ptr GetConstOperandAsTensor(uint32_t index);
    Blob::Ptr/*IRBlob::Ptr*/ GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len);
    void SetOperandMemory(const Model &model, uint32_t index, uint32_t &len_out, const uint8_t *buf);
    void SetOperandFromTensor(uint8_t* buf, uint32_t &length, IRBlob::Ptr infOutput);
//    void SetOperandFromTensor(uint8_t* buf, uint32_t &length, TBlob<float>::Ptr infOutput);
    bool isConst(int index);
    OutputPort getPort(int index);

    Model mModel;
    std::vector<RunTimeOperandInfo> mOperands;
    std::vector<RunTimePoolInfo> mPoolInfos;
    IRDocument mNet;
    std::vector<OutputPort> mPorts;  //typedef std::shared_ptr<Data> DataPtr;
    ExecuteNetwork* enginePtr;
//    std::vector<InferenceEngine::DataPtr> mPorts;

};


}  // namespace vpu_driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif // ANDROID_ML_NN_VPU_PREPAREDMODEL_H
