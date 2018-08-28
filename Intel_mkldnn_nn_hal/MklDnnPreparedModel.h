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

#ifndef ANDROID_ML_NN_MKL_DNN_PREPAREDMODEL_H
#define ANDROID_ML_NN_MKL_DNN_PREPAREDMODEL_H

//#include <android/hardware/neuralnetworks/1.0/IDevice.h>
#include <android/hardware/neuralnetworks/1.0/IPreparedModel.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hidlmemory/mapping.h>
#include <hardware/hardware.h>
#include <mkldnn.hpp>

#include <sys/mman.h>
#include <string>

using ::android::hidl::memory::V1_0::IMemory;

using ::mkldnn::memory;
using ::mkldnn::primitive;
using ::mkldnn::engine;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace mkldnn_driver {

// Information we maintain about each operand during execution that
// may change during execution.
struct RunTimeOperandInfo {
    // The dimensions of the operand follow format
    std::vector<uint32_t> dims;

    memory::data_type type;
    //shape is mkldnn's dims as nchw;
    memory::dims shape;
    memory::format format;

    float scale;
    uint8_t zero;
    void* buffer;
    // The length of the buffer.
    uint32_t length;
    // Whether this is a temporary variable, a model input, a constant, etc.
    OperandLifeTime lifetime;
    // Keeps track of how many operations have yet to make use
    // of this temporary variable.  When the count is decremented to 0,
    // we free the buffer.  For non-temporary variables, this count is
    // always 0.
    uint32_t numberOfUsesLeft;
    memory *pmem;
    std::vector<memory *> stub_pmems;
};

// Used to keep a pointer to each of the memory pools.
struct RunTimePoolInfo {
    sp<IMemory> memory;
    hidl_memory hidlMemory;
    uint8_t* buffer;

    bool set(const hidl_memory& hidlMemory);
    bool update();
};


class MklDnnPreparedModel : public IPreparedModel {
public:
    MklDnnPreparedModel(const Model& model)
          : // Make a copy of the model, as we need to preserve it.
            mModel(model), cpu_engine(nullptr) {}
    ~MklDnnPreparedModel() override {deinitialize();}
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

    bool importOperationConv2D(const Operation& operation);
    bool importOperationPool(const Operation& operation);
    bool importOperationActivation(const Operation& operation);
    bool importOperationConcat(const Operation& operation);
    bool importOperationSoftmax(const Operation& operation);
    bool importOperationLRN(const Operation& operation);
    bool importOperationFC(const Operation& operation);
    bool importOperationAdd(const Operation& operation);

    void initializeInput(RunTimeOperandInfo* input, memory::format format);
    void finalizeOutput(RunTimeOperandInfo* output, memory::format format);
    void addStubPmem(RunTimeOperandInfo* operand, memory* pmem);
    memory* insertReorder(memory* src_mem, memory::format format, memory::data_type type,
                          bool execute, float scale, uint8_t zero = 0);
    memory* insertReorder(memory* src_mem, const memory::desc& desc, bool execute, float scale,
                          uint8_t zero = 0);
    memory* getOperandPmemOfFormatType(const RunTimeOperandInfo& operand, memory::format format,
                                       memory::data_type type);
    memory* getOperandPmemOfDesc(const RunTimeOperandInfo& operand, const memory::desc& desc);
    memory* insertReorderIfNeed(RunTimeOperandInfo* operand, memory::format format,
                                memory::data_type type);
    memory* insertReorderIfNeed(RunTimeOperandInfo* operand, memory::desc desc);
    memory::data_type getOperandNeedType(const RunTimeOperandInfo& operand);
    memory* insertActivation(memory* pmem, FusedActivationFunc activation);

    Model mModel;
    std::vector<RunTimeOperandInfo> mOperands;
    std::vector<RunTimePoolInfo> mPoolInfos;
    std::vector<primitive> mNet;
    engine *cpu_engine;
};

}  // namespace mkldnn_driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif // ANDROID_ML_NN_MKL_DNN_PREPAREDMODEL_H
