#pragma once

#include "IRDocument.h"
#include "IRLayers.h"
#include "ie_plugin_dispatcher.hpp"
#include "inference_engine.hpp"
#include "debug.h"
#ifdef ENABLE_MYRIAD
#include "vpu_plugin_config.hpp"
#endif

#include <android/log.h>
#include <log/log.h>

using namespace IRBuilder;
using namespace InferenceEngine;

template <class T> using  vec = std::vector<T>;

template <typename T>
inline std::ostream & operator << (std::ostream &out, const std::vector<T> &vec) {
    if (vec.empty()) return std::operator<<(out, "[]");
    out << "[" << vec[0];
    for (unsigned i=1; i < vec.size(); i++) {
        out << ", " << vec[i];
    }
    return out << "]";
}

uint32_t getNumberOfElements(const vec<uint32_t> &dims) {
    uint32_t count = 1;
    for (size_t i = 0; i < dims.size(); i++) {
        count *= dims[i];
    }
    return count;
}

TensorDims toDims(const vec<uint32_t> &dims)
{
    TensorDims td;
    for (auto d: dims) td.push_back(d);
    return td;
}
// Function to convert F32 into F16
// F32: exp_bias:127 SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM.
// F16: exp_bias:15  SEEEEEMM MMMMMMMM
#define EXP_MASK_F32 	 0x7F800000U
#define EXP_MASK_F16     0x7C00U


// small helper function to represent uint32_t value as float32
float asfloat(uint32_t v) {
    return *reinterpret_cast<float *>(&v);
}

// Function to convert F32 into F16
float f16tof32(short x) {
    // this is storage for output result
    uint32_t u = x;

    // get sign in 32bit format
    uint32_t s = ((u & 0x8000) << 16);

    // check for NAN and INF
    if ((u & EXP_MASK_F16) == EXP_MASK_F16) {
        // keep mantissa only
        u &= 0x03FF;

        // check if it is NAN and raise 10 bit to be align with intrin
        if (u) {
            u |= 0x0200;
        }

        u <<= (23 - 10);
        u |= EXP_MASK_F32;
        u |= s;
    } else if ((x & EXP_MASK_F16) == 0) {  // check for zero and denormals. both are converted to zero
        u = s;
    } else {
        // abs
        u = (u & 0x7FFF);

        // shift mantissa and exp from f16 to f32 position
        u <<= (23 - 10);

        // new bias for exp (f16 bias is 15 and f32 bias is 127)
        u += ((127 - 15) << 23);

        // add sign
        u |= s;
    }

    // finaly represent result as float and return
    return *reinterpret_cast<float *>(&u);
}

// This function convert f32 to f16 with rounding to nearest value to minimize error
// the denormal values are converted to 0.
short f32tof16(float x) {
    // create minimal positive normal f16 value in f32 format
    // exp:-14,mantissa:0 -> 2^-14 * 1.0
    static float min16 = asfloat((127 - 14) << 23);

    // create maximal positive normal f16 value in f32 and f16 formats
    // exp:15,mantissa:11111 -> 2^15 * 1.(11111)
    static float max16 = asfloat(((127 + 15) << 23) | 0x007FE000);
    static uint32_t max16f16 = ((15 + 15) << 10) | 0x3FF;

    // define and declare variable for intermidiate and output result
    // the union is used to simplify representation changing
    union {
        float f;
        uint32_t u;
    } v;
    v.f = x;

    // get sign in 16bit format
    uint32_t s = (v.u >> 16) & 0x8000;  // sign 16:  00000000 00000000 10000000 00000000

    // make it abs
    v.u &= 0x7FFFFFFF;  // abs mask: 01111111 11111111 11111111 11111111

    // check NAN and INF
    if ((v.u & EXP_MASK_F32) == EXP_MASK_F32) {
        if (v.u & 0x007FFFFF) {
            return s | (v.u >> (23 - 10)) | 0x0200;  // return NAN f16
        } else {
            return s | (v.u >> (23 - 10));  // return INF f16
        }
    }

    // to make f32 round to nearest f16
    // create halfULP for f16 and add it to origin value
    float halfULP = asfloat(v.u & EXP_MASK_F32) * asfloat((127 - 11) << 23);
    v.f += halfULP;

    // if input value is not fit normalized f16 then return 0
    // denormals are not covered by this code and just converted to 0
    if (v.f < min16 * 0.5F) {
        return s;
    }

    // if input value between min16/2 and min16 then return min16
    if (v.f < min16) {
        return s | (1 << 10);
    }

    // if input value more than maximal allowed value for f16
    // then return this maximal value
    if (v.f >= max16) {
        return max16f16 | s;
    }

    // change exp bias from 127 to 15
    v.u -= ((127 - 15) << 23);

    // round to f16
    v.u >>= (23 - 10);

    return v.u | s;
}

void f16tof32Arrays(float *dst, const short *src, uint32_t& nelem, float scale = 1, float bias = 0) {
    ALOGI("convert f16tof32Arrays...\n");
    const short *_src = reinterpret_cast<const short *>(src);

    for (uint32_t i = 0; i < nelem; i++) {
        dst[i] = f16tof32(_src[i]) * scale + bias;
    }
}

void f32tof16Arrays(short *dst, const float *src, uint32_t& nelem, float scale = 1, float bias = 0) {
    ALOGI("convert f32tof16Arrays...");
    for (uint32_t i = 0; i < nelem; i++) {
        dst[i] = f32tof16(src[i] * scale + bias);
        //VLOG(L1, "element no: %d", i);
    }
}

static void setConfig(std::map<std::string, std::string> &config) {
//    config[VPUConfigParams::FIRST_SHAVE] = "0";
//    config[VPUConfigParams::LAST_SHAVE] = "11";
//    config[VPUConfigParams::MEMORY_OPTIMIZATION] = CONFIG_VALUE(NO);//InferenceEngine::PluginConfigParams::YES;
//    config[VPUConfigParams::COPY_OPTIMIZATION] = CONFIG_VALUE(NO);//InferenceEngine::PluginConfigParams::YES;
#ifdef ENABLE_MYRIAD
    config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_INFO);
    config[VPUConfigParams::KEY_VPU_LOG_LEVEL] = CONFIG_VALUE(LOG_DEBUG);
    config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_DEBUG);
#elif ENABLE_MKLDNN
    config[PluginConfigParams::KEY_CPU_BIND_THREAD] = PluginConfigParams::NO;
    config[PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS] = PluginConfigParams::NO;
    config[PluginConfigParams::KEY_DYN_BATCH_LIMIT] = "1";
#endif
    //config[VPU_CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_DEBUG);
    //config[InferenceEngine::PluginConfigParams::CONFIG_KEY(LOG_LEVEL)] = InferenceEngine::PluginConfigParams::LOG_DEBUG;
    //config[VPUConfigParams::VPU_LOG_LEVEL] = CONFIG_VALUE(LOG_DEBUG);
    //config[InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL] = InferenceEngine::PluginConfigParams::LOG_DEBUG /*LOG_WARNING*/;
    //config[InferenceEngine::VPUConfigParams::IGNORE_UNKNOWN_LAYERS] = InferenceEngine::PluginConfigParams::NO;

/*    std::ifstream in("config");
    if (in.is_open()) {
        std::string key, value;
        while (in >> key >> value) {
            config[key] = value;
        }
        in.close();
    }
*/
}


void dumpBlob(const /*std::string*/ char* prefix, size_t len, Blob::Ptr blob)
{
/*
    auto dims = blob->getTensorDesc().getDims();
    std::cout << prefix << dims;

    auto mem = blob->readOnly();

    const short *pf = mem.as<const short*>();

    if (len > blob->size()) len = blob->size();

    for (unsigned i=0; i<len; i++)
    {
        if (0==i % 16)
        {
            std::cout << std::endl<< i<< ": ";
        }
        std::cout << pf[i] << ", ";
    }
    std::cout << std::endl;
*/
}

class ExecuteNetwork
{
    InferenceEnginePluginPtr enginePtr;
    ICNNNetwork *network;
    //IExecutableNetwork::Ptr pExeNet;
    ExecutableNetwork executable_network;
    InputsDataMap inputInfo;
    OutputsDataMap outputInfo;
    IInferRequest::Ptr req;
    InferRequest inferRequest;
    ResponseDesc resp;

public:
    ExecuteNetwork(){}
    ExecuteNetwork(IRDocument &doc, TargetDevice target = TargetDevice::eCPU)
        : network(nullptr)
    {
        InferenceEngine::PluginDispatcher dispatcher({"/vendor/lib64","/vendor/lib","/system/lib64","/system/lib","","./"});
        enginePtr = dispatcher.getSuitablePlugin(target);

        network = doc.getNetwork();
        network->setTargetDevice(target);
        network->getInputsInfo(inputInfo);
        network->getOutputsInfo(outputInfo);
        printf("aks Execute Network intialized\n");
    }

    ExecuteNetwork(ExecutableNetwork& exeNet) : ExecuteNetwork(){
    executable_network = exeNet;
    inferRequest = executable_network.CreateInferRequest();
    ALOGI("aks infer request created");
    printf("aks infer request created\n");
  }

    void loadNetwork()
    {

        std::map<std::string, std::string> networkConfig;
        setConfig(networkConfig);

        printf("Create plugin\n");
        InferencePlugin plugin(enginePtr);
        printf("Plugin load network\n");
        executable_network = plugin.LoadNetwork(*network, networkConfig);
        printf("Plugin loaded network\n");
        std::cout << "Network loaded" << std::endl;

        inferRequest = executable_network.CreateInferRequest();
        std::cout << "infer request created" << std::endl;
      }

    void prepareInput()
    {
      printf("aks prepare Input\n");
      Precision inputPrecision = Precision::FP32;
      inputInfo.begin()->second->setPrecision(inputPrecision);
      inputInfo.begin()->second->setLayout(Layout::NC);

    }

    void prepareOutput()
    {
      printf("aks prepare output\n");
      Precision inputPrecision = Precision::FP32;
      outputInfo.begin()->second->setPrecision(inputPrecision);
      //outputInfo.begin()->second->setLayout(Layout::NC);

      auto dims = inputInfo.begin()->second->getDims();
      printf("input dims size = %d\n", dims.size());
      //outputInfo.begin()->second->setDims(dims);
      auto outputDims = outputInfo.begin()->second->getDims();
      printf("output dims size = %d\n", outputDims.size());
    }

    //for aync infer request
      void setBlob(const std::string& inName, const Blob::Ptr& inputBlob)
      {
          ALOGI("aks setBlob for infer request, input or output name: %s", inName.c_str());
          ALOGI("aks Blob size %d and byte size %d bytes/element %d", inputBlob->size(), inputBlob->byteSize(), inputBlob->element_size());

          //inferRequest.SetBlob(inName.c_str(), inputBlob);
          inferRequest.SetBlob(inName, inputBlob);
          ALOGI("aks infer request set blob done for name %s", inName.c_str());

          //std::cout << "setBlob input or output name : " << inName << std::endl;

      }

     //for non aync infer request
     Blob::Ptr getBlob(const std::string& outName) {
         Blob::Ptr outputBlob;
         outputBlob = inferRequest.GetBlob(outName);
         //std::cout << "GetBlob input or output name : " << outName << std::endl;
         ALOGI("aks GetBlob input or output name : ", outName.c_str());
         return As<TBlob<short>>(outputBlob);
     }

    TBlob<float>::Ptr Infer(const Blob::Ptr &in)
    {
//        printf("infer network\n");
//        inferRequest = executable_network.CreateInferRequest();

        auto inName = inputInfo.begin()->first;
        printf("set input blob\n");
        inferRequest.SetBlob(inName, in);

        printf("aks prepare output blob\n");
        const std::string firstOutName = outputInfo.begin()->first;
        InferenceEngine::TBlob<PrecisionTrait<Precision::FP32>::value_type>::Ptr outputBlob;
        outputBlob = InferenceEngine::make_shared_blob<PrecisionTrait<Precision::FP32>::value_type,
                InferenceEngine::SizeVector>(Precision::FP32, outputInfo.begin()->second->getDims());
        outputBlob->allocate();

        printf("set output blob\n");
        inferRequest.SetBlob(firstOutName, outputBlob);

        //inferRequest = executable_network.CreateInferRequest();
        printf("aks StartAsync triggered\n");
        inferRequest.StartAsync();  //for async infer
        printf("aks wait triggered");
        inferRequest.Wait(1000);

        std::cout << "output name : " << firstOutName << std::endl;
        ALOGI("aks output name : %s",firstOutName.c_str());
        //Blob::Ptr outBlob;
        //req->GetBlob(firstOutName.c_str(), outBlob, &resp);
        return As<TBlob<float>>(outputBlob);
    }
};

/*
class InfEng
{
    InferenceEnginePluginPtr enginePtr;

public:

    InfEng(TargetDevice target = TargetDevice::eCPU)
    {
        ResponseDesc resp;
        InferenceEngine::PluginDispatcher dispatcher({"/vendor/lib64","/vendor/lib","/system/lib64","/system/lib","","./"});
        enginePtr = dispatcher.getSuitablePlugin(target);
    }


    ExeNet Load(IRDocument &doc)
    {
        ResponseDesc resp;
//        ExeNet ret;
        auto network = doc.getNetwork();
        //network.setBatchSize(1);


        std::map<std::string, std::string> networkConfig;
        setConfig(networkConfig);
        //const auto rc = enginePtr->LoadNetwork(pExec, *network, networkConfig, &resp);
        InferencePlugin plugin(enginePtr);
        auto executable_network = plugin.LoadNetwork(*network, networkConfig);

        //ExeNet ret(pExec);
        ExeNet ret(executable_network);
        ret.inputInfo(network->getInputsInfo());
        ret.outputInfo(network->getOutputsInfo());
        ALOGI("aks Network loaded");
        printf("aks Network loaded\n");



        std::cout << "Network loaded" << std::endl;
        return ret;
    }

};
*/
