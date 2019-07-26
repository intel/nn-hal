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

#include <android/hardware/neuralnetworks/1.0/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.1/types.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hidlmemory/mapping.h>
#include <hardware/hardware.h>
#include <sys/mman.h>
#include <string>
#include <fstream>
#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>
#include <thread>
#include "IENetwork.h"
#include "Driver.h"

using ::android::hidl::memory::V1_0::IMemory;
using namespace InferenceEngine;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

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
class PreparedModel : public IPreparedModel {
public:
    PreparedModel(const Model& model)
          :mTargetDevice(TargetDevice::eMYRIAD), mModel(model), mNet("nnNet"), enginePtr(nullptr) {
        g_layer_precision = InferenceEngine::Precision::FP16;
    }

    PreparedModel(const TargetDevice device, const Model& model)
          :mTargetDevice(device), mModel(model), mNet("nnNet"), enginePtr(nullptr) {
        if (mTargetDevice == TargetDevice::eCPU)
           g_layer_precision = InferenceEngine::Precision::FP32;
        else if (mTargetDevice == TargetDevice::eMYRIAD)
           g_layer_precision = InferenceEngine::Precision::FP16;
        else
           g_layer_precision = InferenceEngine::Precision::UNSPECIFIED;
    }

    ~PreparedModel() override {deinitialize();}
    bool initialize();
    Return<ErrorStatus> execute(const Request& request,
                                const sp<IExecutionCallback>& callback) override;
    static bool isOperationSupported(const Operation& operation, const Model& model);

protected:
    void deinitialize();
    bool initializeRunTimeOperandInfo();
    void asyncExecute(const Request& request, const sp<IExecutionCallback>& callback);

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
    bool operationMean(const Operation& operation);

    void initializeInput();
    bool finalizeOutput(/*RunTimeOperandInfo* output*/);

    OutputPort handleFusion(const OutputPort &out, int32_t fusedOp);
    template<typename T>
    T GetConstFromBuffer(const uint8_t *buf, uint32_t len);
    template<typename T>
    std::vector<T> GetConstVecFromBuffer(const uint8_t *buf, uint32_t len);
    const uint8_t *GetOperandMemory(const Model &model, uint32_t index, uint32_t &len_out);
    template <typename T>
    T ParseOperationInput(const Model &model, const Operation& operation, uint32_t index);
    template <typename T>
    T GetConstOperand(const Model &model, uint32_t index);
    template <typename T>
    std::vector<T> GetConstVecOperand(const Model &model, uint32_t index);
    virtual Blob::Ptr GetConstOperandAsTensor(int operand_index,int operation_idx);
    virtual Blob::Ptr GetInOutOperandAsBlob(bool param_type,RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len);
    virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index);
    void SetOperandMemory(const Model &model, uint32_t index, uint32_t &len_out, const uint8_t *buf);
    void SetOperandFromTensor(uint8_t* buf, uint32_t &length, Blob::Ptr infOutput);
    bool isConst(int index);
    OutputPort getPort(int index);

    TargetDevice mTargetDevice;
    Model mModel;
    std::vector<RunTimeOperandInfo> mOperands;
    std::vector<RunTimePoolInfo> mPoolInfos;
    IRDocument mNet;
    std::vector<OutputPort> mPorts;  //typedef std::shared_ptr<Data> DataPtr;
    ExecuteNetwork* enginePtr;
    bool mPadreq;
};

class VpuPreparedModel : public PreparedModel {
public:
    VpuPreparedModel(const Model& model)
          :PreparedModel(TargetDevice::eMYRIAD, model) {
    }

    virtual Blob::Ptr GetConstOperandAsTensor(int operand_index,int operation_idx) override;
    virtual Blob::Ptr GetInOutOperandAsBlob(bool param_type,RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len) override;
    virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index) override;
};

class CpuPreparedModel : public PreparedModel {
public:
    CpuPreparedModel(const Model& model)
          :PreparedModel(TargetDevice::eCPU, model) {
    }

    virtual Blob::Ptr GetConstOperandAsTensor(int operand_index,int operation_idx) override;
    virtual Blob::Ptr GetInOutOperandAsBlob(bool param_type,RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len) override;
    virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index) override;
};

#define DISABLE_ALL_QUANT
//#define NN_DEBUG

enum DebugLevel {
    L0,
    L1,
    L2,
    L3,
    L4,
};

#define DebugMask(Level)  ((1 << (Level + 1)) - 1)

#ifdef NN_DEBUG

#define VLOG(l, x, ...)                                                \
    do {                                                               \
        if (DebugMask(l) & (1 << l))                                      \
            ALOGI("[%s] " x, __FUNCTION__, ##__VA_ARGS__);             \
    } while(0)

#define VLOGDIMS(l, d, header)                                         \
    do {                                                               \
        auto size = (d).size();                                        \
        VLOG(l, "%s: vectors {%d, %d, %d, %d}",                        \
                 header,size > 0 ? (d)[0] : 0, size > 1 ? (d)[1] : 0,                 \
                size > 2 ? (d)[2] : 0, size > 3 ? (d)[3] : 0);         \
    } while(0)

#define dumpOperand(index)                                             	\
    do {                                                               	\
        const auto op = mModel.operands[index];                         \
        ALOGI("dump:%s,Operand-index:%d",__func__,index);         \
        ALOGI("%s", toString(op).c_str());                              \
        ALOGI("---------------------------------------------");         \
    } while (0)
#define dumpOp(op,index)                                             	\
    do {                                                               	\
        ALOGI("[%s] dumping Operand_index %d",__func__,index);         \
        ALOGI("%s", toString(op).c_str());                              \
        ALOGI("---------------------------------------------");         \
    } while (0)
		
#define dumpOperation(operation)                                        \
    do {                                                                \
        ALOGI("---------------------------------------------");         \
        ALOGI("[%s] dumping Operation :",__func__);         \
        ALOGI("%s", toString(operation).c_str());                       \
        ALOGI("---------------------------------------------");         \
    } while (0)

#define dumpOperationParam(operation)                        \
    do {                                                                \
        ALOGI("dumping operation-params");         \
        ALOGI("%s", toString(operation).c_str());                       \
    } while (0)

#else
#define VLOG(...)
#define VLOGDIMS(l, d, header)
#define dumpOperand(...)
#define dumpOperation(operation)
#define dumpOperationSupport(operation, support)
#define dumpOp(op,index)  
#define dumpOperationParam(operation)  
#endif

#define WRONG_DIM  (-1)

#define nnAssert(v)                                                                          \
    do {                                                                                       \
        if (!(v)) {                                                                            \
            LOG(ERROR) << "nnAssert failed at " << __FILE__ << ":" << __LINE__ << " - '" << #v \
                       << "'\n";                                                               \
					    }                                                                                      \
    } while (0)

#define PARAM_I32(i) ParseOperationInput<int32_t>(mModel, operation, i)
#define PARAM_FP(i) ParseOperationInput<float>(mModel, operation, i)
		
#define EXPL_PAD_PARAMS_CONV 10
#define IMPL_PAD_PARAMS_CONV 7
#define IN_PARAM 1
#define OUT_PARAM 2
#define EXPL_PAD_PARAMS_DW_CONV 11
#define IMPL_PAD_PARAMS_DW_CONV 8
#define SOFTMAX_INPUT_PARAMS 2
#define EXPL_PAD 1 
#define IMPL_PAD 2
#define NHWC_DIM_NUM 4
#define NHWC_CH_IDX 3
#define NHWC_HT_IDX 1
#define NHWC_WD_IDX 2
//operand index as from  1.1/type.hal 
#define OP_INPUT_IDX_CONV 0
#define OP_FILTER_IDX_CONV 1
#define OP_BIAS_IDX_CONV 2
#define OP_PADSCHEME_IDX_CONV 3
#define OP_PADL_IDX_CONV 3
#define OP_PADR_IDX_CONV 4
#define OP_PADH_IDX_CONV 5
#define OP_PADW_IDX_CONV 6
#define OP_STRD_WD_IDX_EXPL_CONV  7
#define OP_STRD_HT_IDX_EXPL_CONV  8
#define OP_STRD_WD_IDX_IMPL_CONV  4
#define OP_STRD_HT_IDX_IMPL_CONV  5
#define OP_ACTV_FUNC_IDX_IMPL_CONV  6 
#define OP_ACTV_FUNC_IDX_EXPL_CONV  9 
#define OP_ACTV_FUNC_IDX_IMPL_DW_CONV  7 
#define OP_ACTV_FUNC_IDX_EXPL_DW_CONV  10
#define OP_DW_CONV_DPM_IMPL 6  //depth multiplier 
#define OP_DW_CONV_DPM_EXPL 9
#define OP_ADD_OPR1_IDX 0
#define OP_ADD_OPR1_IDX 1

//average_pooling_2d as in type.hal 
#define EXPL_PAD_PARAMS_POOL 10
#define IMPL_PAD_PARAMS_POOL 7
#define OP_INPUT_IDX_POOL 0
#define OP_PADL_IDX_POOL 1
#define OP_PADR_IDX_POOL 2
#define OP_PADH_IDX_POOL 3
#define OP_PADW_IDX_POOL 4
#define OP_STRD_WD_IDX_EXPL_POOL  5
#define OP_STRD_HT_IDX_EXPL_POOL  6
#define OP_FLT_WD_IDX_EXPL_POOL 7
#define OP_FLT_HT_IDX_EXPL_POOL 8
#define OP_ACTV_FUNC_IDX_EXPL_POOL  9
 
#define OP_PADSCHEME_IDX_POOL 1
#define OP_STRD_WD_IDX_IMPL_POOL  2
#define OP_STRD_HT_IDX_IMPL_POOL  3
#define OP_FLT_WD_IDX_IMPL_POOL 4
#define OP_FLT_HT_IDX_IMPL_POOL 5
#define OP_ACTV_FUNC_IDX_IMPL_POOL  6 

//fully_connected as in type.hal
#define OP_INPUT_IDX_FC 0 
#define OP_WGHT_IDX_FC 1
#define OP_BIAS_IDX_FC 2
#define OP_ACTV_IDX_FC 3 
#define FC_INPUT_PARAMS 4

//ADD operation
#define ADD_INPUT_PARAMS 3
#define OP_INPUT0_IDX_ADD 0
#define OP_INPUT1_IDX_ADD  1
#define OP_ACTV_IDX_ADD 2

#define CHECK_OPERAND_2D(params,idx_x,idx_y) \
		do{\
			VLOG(L1,"As found in %s",__func__); \
			if (params.x<0 || params.y<0) {\
				 VLOG(L1,"Invalid Point2D Operands at index [%d ,%d] , aborting!!",idx_x,idx_y); \
				 return false;\
			}\
		}while(0)	

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android 

#endif // ANDROID_ML_NN_PREPAREDMODEL_H
