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

#define LOG_TAG "MklDnnPreparedModel"

#include <android-base/logging.h>
#include <cutils/log.h>
#include <thread>

#include "MklDnnPreparedModel.h"

#define DISABLE_ALL_QUANT

enum MklDnnDebugLevel {
    L0,
    L1,
    L2,
    L3,
    L4,
};

//#define MKLDNN_DEBUG

unsigned int debugMask = ((1 << (L2 + 1)) - 1);

#ifdef MKLDNN_DEBUG
#define VLOG(l, x, ...)                                                \
    do {                                                               \
        if (debugMask & (1 << l))                                      \
            ALOGI("[%s] " x, __FUNCTION__, ##__VA_ARGS__);             \
    } while(0)

#define VLOGDIMS(l, d, header)                                         \
    do {                                                               \
        auto size = (d).size();                                        \
        VLOG(l, "%s: vectors {%d, %d, %d, %d}",                        \
                 header,(d)[0], size > 1 ? (d)[1] : 0,                 \
                size > 2 ? (d)[2] : 0, size > 3 ? (d)[3] : 0);         \
    } while(0)

#else
#define VLOG(...)
#define VLOGDIMS(l, d, header)
#endif

#define MKLDNN_WRONG_DIM  (-1)

#define nnAssert(v)                                                                            \
    do {                                                                                       \
        if (!(v)) {                                                                            \
            LOG(ERROR) << "nnAssert failed at " << __FILE__ << ":" << __LINE__ << " - '" << #v \
                       << "'\n";                                                               \
            abort();                                                                           \
        }                                                                                      \
    } while (0)

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace mkldnn_driver {

enum PaddingScheme {
    kPaddingUnknown = 0,
    kPaddingSame = 1,
    kPaddingValid = 2,
};

void calculateExplicitPadding(int32_t in_size, int32_t stride,
                              int32_t filter_size, int32_t padding_implicit,
                              int32_t* padding_head, int32_t* padding_tail) {
    *padding_head = 0;
    *padding_tail = 0;

    if (padding_implicit == kPaddingSame) {
        int32_t out_size = (in_size + stride - 1) / stride;
        int32_t tmp = (out_size - 1) * stride + filter_size;
        if (tmp > in_size) {
            *padding_head = (tmp - in_size) / 2;
            *padding_tail = (tmp - in_size) - *padding_head;
        }
    }
}

int32_t computeOutSize(int32_t imageSize, int32_t filterSize, int32_t stride,
                               int32_t paddingHead, int32_t paddingTail) {
    return (imageSize - filterSize + stride + paddingHead + paddingTail) / stride;
}

//shape is nchw, dims depends on format
bool dimsToShape(const std::vector<uint32_t>& dims, memory::format format, memory::dims* shape)
{

    VLOG(L3, "format: %d", static_cast<int>(format));
    VLOGDIMS(L3, dims, "dims");
    int n, c, h, w;
    //4-D
    switch (format) {
        case memory::format::nchw:
        case memory::format::oihw:
        case memory::format::nChw8c:
        case memory::format::nChw16c:
            n = dims[0];
            c = dims[1];
            h = dims[2];
            w = dims[3];
            *shape = {n, c, h, w};
            break;
        case memory::format::nhwc:
        case memory::format::Ohwi8o:
        case memory::format::Ohwi16o:
            n = dims[0];
            h = dims[1];
            w = dims[2];
            c = dims[3];
            *shape = {n, c, h, w};
            break;
        case memory::format::ihwo:
            n = dims[3];
            c = dims[0];
            h = dims[1];
            w = dims[2];
            *shape = {n, c, h, w};
            break;
        case memory::format::x:
            n = dims[0];
            *shape = {n};
            break;
        case memory::format::nc:
            n = dims[0];
            c = dims[1];
            *shape = {n, c};
            break;
        default:
            return false;
    }

    VLOGDIMS(L3, *shape, "shape");
    return true;
}

//shape is nchw, dims depends on format
bool shapeToDims(const memory::dims& shape, std::vector<uint32_t>* dims, memory::format format)
{

    VLOG(L3, "format: %d", static_cast<int>(format));
    VLOGDIMS(L3, shape, "shape");
    uint32_t n, c, h, w;
    //1-D
    if (format == memory::format::x) {
            n = shape[0];
            *dims = {n};
            return true;
    }

    if (format == memory::format::nc) {
            n = shape[0];
            c = shape[1];
            *dims = {n, c};
            return true;
    }


    //4-D
    //mkldnn accept nchw or oihw.
    n = shape[0];
    c = shape[1];
    h = shape[2];
    w = shape[3];

    switch (format) {
        case memory::format::nchw:
        case memory::format::oihw:
        case memory::format::nChw8c:
        case memory::format::nChw16c:
           *dims = {n, c, h, w};
            break;
        case memory::format::nhwc:
        case memory::format::Ohwi8o:
        case memory::format::Ohwi16o:
            *dims = {n, h, w, c};
            break;
        case memory::format::ihwo:
            *dims = {c, h, w, n};
            break;
        default:
            return false;
    }

    VLOGDIMS(L3, *dims, "dims");
    return true;
}

inline size_t getSizeFromInts(int lower, int higher)
{
    return (uint32_t)(lower) + ((uint64_t)(uint32_t)(higher) << 32);
}

template <typename T>
T getScalarData(const RunTimeOperandInfo& info)
{
    // TODO: Check buffer is at least as long as size of data.
    T* data = reinterpret_cast<T*>(info.buffer);
    return data[0];
}

int sizeOfData(memory::data_type type, std::vector<uint32_t> dims)
{
    int size;
    switch(type) {
        case memory::data_type::f32:
            size = 4;
            break;
        case memory::data_type::s32:
            size = 4;
            break;
        case memory::data_type::s16:
            size = 2;
            break;
        case memory::data_type::u8:
        case memory::data_type::s8:
            size = 1;
            break;
        default:
            size = 0;
    }
    for (auto d : dims)
        size *= d;

    return size;
}

bool RunTimePoolInfo::set(const hidl_memory& hidlMemory) {
    this->hidlMemory = hidlMemory;
    auto memType = hidlMemory.name();
    if (memType == "ashmem") {
        memory = mapMemory(hidlMemory);
        if (memory == nullptr) {
            ALOGE("Can't map shared memory.");
            return false;
        }
        memory->update();
        buffer = reinterpret_cast<uint8_t*>(static_cast<void*>(memory->getPointer()));
        if (buffer == nullptr) {
            ALOGE("Can't access shared memory.");
            return false;
        }
        return true;
    } else if (memType == "mmap_fd") {
        size_t size = hidlMemory.size();
        int fd = hidlMemory.handle()->data[0];
        int prot = hidlMemory.handle()->data[1];
        size_t offset = getSizeFromInts(hidlMemory.handle()->data[2],
                                        hidlMemory.handle()->data[3]);
        buffer = static_cast<uint8_t*>(mmap(nullptr, size, prot, MAP_SHARED, fd, offset));
        if (buffer == MAP_FAILED) {
            ALOGE("Can't mmap the file descriptor.");
            return false;
        }
        return true;
    } else {
        ALOGE("unsupported hidl_memory type");
        return false;
    }
}

// Making sure the output data are correctly updated after execution.
bool RunTimePoolInfo::update() {
    auto memType = hidlMemory.name();
    if (memType == "ashmem") {
        memory->commit();
        return true;
    } else if (memType == "mmap_fd") {
        int prot = hidlMemory.handle()->data[1];
        if (prot & PROT_WRITE) {
            size_t size = hidlMemory.size();
            return msync(buffer, size, MS_SYNC) == 0;
        }
    }
    // No-op for other types of memory.
    return true;
}

bool setRunTimePoolInfosFromHidlMemories(std::vector<RunTimePoolInfo>* poolInfos,
                                         const hidl_vec<hidl_memory>& pools) {
    poolInfos->resize(pools.size());
    for (size_t i = 0; i < pools.size(); i++) {
        auto& poolInfo = (*poolInfos)[i];
        if (!poolInfo.set(pools[i])) {
            ALOGE("Could not map pool");
            return false;
        }
    }
    return true;
}

//init from {buffer, type, format} to {pmem, shape}
void MklDnnPreparedModel::initializeInput(RunTimeOperandInfo* input, memory::format format)
{
    VLOG(L2, "input pmem %p", input->pmem);
    //already inited;
    if (input->pmem != nullptr) {
        return;
    }

    bool is_model_input = (input->lifetime == OperandLifeTime::MODEL_INPUT);
    nnAssert(is_model_input || input->buffer != nullptr);
    nnAssert(input->stub_pmems.size() == 0);
    nnAssert(input->format == memory::format::format_undef);
    input->format = format;

    nnAssert(dimsToShape(input->dims, input->format, &input->shape) == true);

    VLOG(L2, "new memory primitive of type %d, format %d, shape: %d, %d, %d, %d",
            input->type, input->format, input->shape[0],
            input->shape.size() > 1 ? input->shape[1] : 0,
            input->shape.size() > 2 ? input->shape[2] : 0,
            input->shape.size() > 3 ? input->shape[3] : 0);

    auto primitive_desc_mem = memory::primitive_desc({input->shape, input->type,
                                            input->format}, *cpu_engine);
    if (input->buffer == nullptr) {
        VLOG(L2, "input buffer is null");
        input->pmem = new memory(primitive_desc_mem);
        input->buffer = input->pmem->get_data_handle();
    } else {
        VLOG(L2, "input buffer is %p", input->buffer);
        input->pmem = new memory(primitive_desc_mem, input->buffer);
    }
    VLOG(L2, "get new memory %p, buffer at %p", input->pmem, input->buffer);
}

//from {pmem, shape} to {new pmem, type, buffer, format, length}
void MklDnnPreparedModel::finalizeOutput(RunTimeOperandInfo* output, memory::format format)
{
    nnAssert(output->pmem != nullptr);
    nnAssert(output->stub_pmems.size() == 0);

    VLOG(L2, "output pmem is %p", output->pmem);
    bool is_model_output = (output->lifetime == OperandLifeTime::MODEL_OUTPUT);
    if (is_model_output) {
        //request output is nhwc
        auto src_pmem = output->pmem;
        auto desc_src = src_pmem->get_primitive_desc().desc();
        auto format_src = static_cast<memory::format>(desc_src.data.format);
        auto type_src = static_cast<memory::data_type>(desc_src.data.data_type);
        VLOG(L2, "output is model output");
        if (format_src != format || type_src != output->type) {
            VLOG(L2, "output need reorder from format:type %d:%d to %d:%d",
                     format_src, type_src, format, output->type);
            //may scale f32 to u8
            float scale = output->scale ? 1 / output->scale : 0;
            output->pmem = insertReorder(src_pmem, format, output->type, false, scale);
            if (output->pmem != src_pmem) {
                VLOG(L2, "output new pmem is %p", output->pmem);
                addStubPmem(output, src_pmem);
            }
        }
    } else {
        //for temporary variabile, skip the quant. Do not dequant.
        VLOG(L2, "output is temporary variables");
        output->scale = 0;
        output->zero = 0;
    }

    auto md_output = output->pmem->get_primitive_desc().desc();

    output->format = static_cast<memory::format>(md_output.data.format);
    nnAssert(shapeToDims(output->shape, &output->dims, output->format));
    output->type = static_cast<memory::data_type>(md_output.data.data_type);
    output->buffer = output->pmem->get_data_handle();
    //in case output length changes
    output->length = sizeOfData(output->type, output->dims);
    VLOG(L2, "set output attr format %d, type %d, buffer %p, length %d",
             output->format, output->type, output->buffer, output->length);
}

//return the needed type for operation
memory::data_type MklDnnPreparedModel::getOperandNeedType(const RunTimeOperandInfo& operand)
{
    if (operand.scale != 0) {
        return memory::data_type::f32;
    }

    return operand.type;
}

memory* MklDnnPreparedModel::getOperandPmemOfFormatType(const RunTimeOperandInfo& operand,
                                              memory::format format, memory::data_type type)
{
    VLOG(L2, "on operand of pmem %p, search format %d, type %d", operand.pmem, format, type);
    if (operand.format == format && (type == memory::data_undef || operand.type == type)) {
        //if operand not init, pmem is null
        return operand.pmem;
    }

    for (const auto& pmem : operand.stub_pmems) {
        if (pmem->get_primitive_desc().desc().data.format == format &&
            (type == memory::data_undef ||
            pmem->get_primitive_desc().desc().data.data_type == type)) {
            VLOG(L2, "get it %p at stub_pmems", pmem);
            return pmem;
        }
    }
    VLOG(L2, "No pmem found");
    return nullptr;
}

memory* MklDnnPreparedModel::getOperandPmemOfDesc(const RunTimeOperandInfo& operand,
                                                  const memory::desc& desc)
{
    auto format = static_cast<memory::format>(desc.data.format);
    auto type = static_cast<memory::data_type>(desc.data.data_type);
    return getOperandPmemOfFormatType(operand, format, type);
}


void MklDnnPreparedModel::addStubPmem(RunTimeOperandInfo* operand, memory* pmem)
{
    VLOG(L2, "add pmem %p to operand %p stub pmems", pmem, operand);
    operand->stub_pmems.push_back(pmem);
}

memory* MklDnnPreparedModel::insertActivation(memory* pmem, FusedActivationFunc activation)
{
    VLOG(L2, "insert activation of %d to pmem %p", activation, pmem);
    mkldnn::algorithm alg;
    float alpha = 0;
    switch(activation) {
        case FusedActivationFunc::RELU:
            alg = mkldnn::algorithm::eltwise_relu;
            break;
        case FusedActivationFunc::RELU6:
            alg = mkldnn::algorithm::eltwise_bounded_relu;
            alpha = 6;
            break;
        default:
            nnAssert(false);
            return nullptr;
    }

    auto pmem_output = new memory(pmem->get_primitive_desc());
    auto desc_relu = mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward,
                        alg, pmem->get_primitive_desc().desc(), alpha);
    auto primitive_desc_relu = mkldnn::eltwise_forward::primitive_desc(desc_relu,
                        *cpu_engine);

    mNet.push_back(mkldnn::eltwise_forward(primitive_desc_relu, *pmem,
                        *pmem_output));

    return pmem_output;
}

//insert reorder depends on mem_pd, if not return src_mem
memory* MklDnnPreparedModel::insertReorder(memory* src_mem, memory::format format,
                                           memory::data_type type, bool execute, float scale, uint8_t zero)
{
    VLOG(L2, "insert reorder for format %d, type %d, scale %f, zero %d", format, type, scale, zero);
    auto desc_src = src_mem->get_primitive_desc().desc();
    //we only compare format and type
    auto format_src = static_cast<memory::format>(desc_src.data.format);
    auto type_src = static_cast<memory::data_type>(desc_src.data.data_type);
    VLOG(L2, "insert reorder src format %d, type %d", format_src, type_src);
    if (format_src == format && type_src == type) {
        return src_mem;
    }

    memory::dims shape;
    shape.resize(desc_src.data.ndims);
    for (int i = 0; i < desc_src.data.ndims; i++) {
        shape[i] = desc_src.data.dims[i];
    }
    memory* dst_mem = new memory({{shape, type, format}, *cpu_engine});
    if (scale != 0) {
        VLOG(L2, "reorder need scale");
        mkldnn::primitive_attr attr;
        attr.set_output_scales(0, {scale});
        attr.set_int_output_round_mode(mkldnn::round_nearest);
        auto pd_reorder = mkldnn::reorder::primitive_desc(src_mem->get_primitive_desc(),
                                                   dst_mem->get_primitive_desc(), attr);
        if (execute) {
            std::vector<primitive> net;
            net.push_back(mkldnn::reorder(pd_reorder, *src_mem, *dst_mem));
            mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
        } else {
            mNet.push_back(mkldnn::reorder(pd_reorder, *src_mem, *dst_mem));
        }
    } else {
        if (execute) {
            std::vector<primitive> net;
            net.push_back(mkldnn::reorder(*src_mem, *dst_mem));
            mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
        } else {
            mNet.push_back(mkldnn::reorder(*src_mem, *dst_mem));
        }
    }

#ifdef MKLDNN_DEBUG
    {
        auto logDesc = [&](const memory *pmem, std::string header) {
            auto desc = pmem->get_primitive_desc().desc();
            auto format = desc.data.format;
            auto type = desc.data.data_type;
            VLOG(L2, "%s has format %d type %d", header.c_str(), format, type);
        };
        logDesc(src_mem, "before reorder src desc");
        logDesc(dst_mem, "after reorder dst desc");
    }
#endif

    return dst_mem;
}

//insert reorder depends on mem_pd, if not return src_mem
memory* MklDnnPreparedModel::insertReorder(memory* src_mem, const memory::desc& desc,
                                           bool execute, float scale, uint8_t zero)
{
    auto format = static_cast<memory::format>(desc.data.format);
    auto type = static_cast<memory::data_type>(desc.data.data_type);
    return insertReorder(src_mem, format, type, execute, scale, zero);
}

//query {format, type} primitive, if not, allocate one, and order to it
memory* MklDnnPreparedModel::insertReorderIfNeed(RunTimeOperandInfo* operand,
                                                 memory::format format,
                                                 memory::data_type type)
{
    bool execute = false;
    auto pmem = getOperandPmemOfFormatType(*operand, format, type);
    if (pmem) {
        return pmem;
    }

    VLOG(L2, "operand lifetime is %d", operand->lifetime);
    if (operand->lifetime == OperandLifeTime::CONSTANT_COPY
        || operand->lifetime == OperandLifeTime::CONSTANT_REFERENCE)
        execute = true;

    pmem = insertReorder(operand->pmem, format, type, execute, operand->scale, operand->zero);
    addStubPmem(operand, pmem);

    return pmem;
}

//query {desc} primitive, if not, allocate one, and order to it
memory* MklDnnPreparedModel::insertReorderIfNeed(RunTimeOperandInfo* operand,
                                                 memory::desc desc)
{
    bool execute = false;
    auto pmem = getOperandPmemOfDesc(*operand, desc);
    if (pmem) {
        return pmem;
    }

    VLOG(L2, "operand lifetime is %d", operand->lifetime);
    if (operand->lifetime == OperandLifeTime::CONSTANT_COPY
        || operand->lifetime == OperandLifeTime::CONSTANT_REFERENCE)
        execute = true;

    pmem =  insertReorder(operand->pmem, desc, execute, operand->scale, operand->zero);
    addStubPmem(operand, pmem);

    return pmem;
}

bool MklDnnPreparedModel::importOperationConv2D(const Operation& operation)
{
    const hidl_vec<uint32_t>& ins = operation.inputs;
    const hidl_vec<uint32_t>& outs = operation.outputs;
    const size_t in_counts = ins.size();
    bool group = (operation.type == OperationType::DEPTHWISE_CONV_2D);

    VLOG(L1, "import %s has inputs %zu", group ? "DEPTHWISE_CONV2D" : "CONV2D", in_counts);
    if (group) {
        if (in_counts != 11 && in_counts != 8) {
            return false;
        }
    } else {
        if (in_counts != 10 && in_counts != 7) {
            return false;
        }
    }

    RunTimeOperandInfo& input = mOperands[ins[0]];
    RunTimeOperandInfo& filter = mOperands[ins[1]];
    RunTimeOperandInfo& bias = mOperands[ins[2]];
    RunTimeOperandInfo& output = mOperands[outs[0]];

    initializeInput(&input, memory::format::nhwc);
    if (group) {
        //depthwise: mkldnn use oihw as shape, format is ohwi.
        initializeInput(&filter, memory::format::ihwo);
    } else {
        //mkldnn use nchw(oihw) as shape, format is nhwc(ohwi).
        initializeInput(&filter, memory::format::nhwc);
    }
    initializeInput(&bias, memory::format::x);

    VLOGDIMS(L2, input.dims, "input has dims");
    VLOGDIMS(L2, input.shape, "input has shape");

    int32_t batches = input.shape[0];
    int32_t channels = input.shape[1];
    int32_t input_height = input.shape[2];
    int32_t input_width = input.shape[3];

    int32_t filter_out = filter.shape[0];
    int32_t filter_in = filter.shape[1];
    int32_t filter_height = filter.shape[2];
    int32_t filter_width = filter.shape[3];

    int32_t padding_left, padding_right;
    int32_t padding_top, padding_bottom;
    int32_t stride_width, stride_height;
    int32_t depth_multiplier = 0;
    FusedActivationFunc activation;

    bool explicit_padding = group ? (in_counts == 11) : (in_counts == 10);
    if (explicit_padding) {
        padding_left     = getScalarData<int32_t>(mOperands[ins[3]]);
        padding_right    = getScalarData<int32_t>(mOperands[ins[4]]);
        padding_top      = getScalarData<int32_t>(mOperands[ins[5]]);
        padding_bottom   = getScalarData<int32_t>(mOperands[ins[6]]);
        stride_width     = getScalarData<int32_t>(mOperands[ins[7]]);
        stride_height    = getScalarData<int32_t>(mOperands[ins[8]]);
        if (group) {
            depth_multiplier = getScalarData<int32_t>(mOperands[ins[9]]);
            activation       = getScalarData<FusedActivationFunc>(mOperands[ins[10]]);
        } else {
            activation       = getScalarData<FusedActivationFunc>(mOperands[ins[9]]);
        }
    } else {
        int32_t padding_implicit = getScalarData<int32_t>(mOperands[ins[3]]);
        stride_width     = getScalarData<int32_t>(mOperands[ins[4]]);
        stride_height    = getScalarData<int32_t>(mOperands[ins[5]]);
        if (group) {
            depth_multiplier = getScalarData<int32_t>(mOperands[ins[6]]);
            activation       = getScalarData<FusedActivationFunc>(mOperands[ins[7]]);
        } else {
            activation       = getScalarData<FusedActivationFunc>(mOperands[ins[6]]);
        }

        calculateExplicitPadding(input_width, stride_width,
                                 filter_width, padding_implicit,
                                 &padding_left, &padding_right);
        calculateExplicitPadding(input_height, stride_height,
                                 filter_height, padding_implicit,
                                 &padding_top, &padding_bottom);
    }

    if (group) {
        nnAssert(filter_out == channels * depth_multiplier);
    } else {
        nnAssert(channels == filter_in);
    }

    VLOG(L2, "batches %d, channels %d, input_height: %d, input_width %d",
             batches, channels, input_height, input_width);
    VLOG(L2, "channels_in %d, channels_out %d, filter_height: %d, filter_width %d",
             filter_in, filter_out, filter_height, filter_width);
    VLOG(L2, "depth multiplier %d", depth_multiplier);
    VLOG(L2, "padding: top: %d, left %d, bottom %d, right %d", padding_top, padding_left, padding_bottom, padding_right);
    VLOG(L2, "stride: height %d, width %d", stride_height, stride_width);
    int32_t output_height = computeOutSize(input_height, filter_height, stride_height,
                                        padding_top, padding_bottom);
    int32_t output_width = computeOutSize(input_width, filter_width, stride_width,
                                       padding_left, padding_right);

    //get output shape, mkldnn define shape as nchw
    output.shape = {batches, filter_out, output_height, output_width};

    auto type_conv_input = getOperandNeedType(input);
    auto type_conv_filter = getOperandNeedType(filter);
    auto type_conv_bias = getOperandNeedType(bias);
    auto type_conv_output = type_conv_input;

    VLOG(L2, "conv types: input %d -> %d, filter %d -> %d, bias %d -> %d",
             input.type, type_conv_input, filter.type, type_conv_filter, bias.type, type_conv_bias);
    if (group) {
        auto format_filter_group = memory::format::goihw;
        auto pmem_group_filter = getOperandPmemOfFormatType(filter, format_filter_group,
                                                            type_conv_filter);
        if (pmem_group_filter == nullptr) {
            auto pmem = insertReorderIfNeed(&filter, memory::format::oihw, type_conv_filter);
            /*
            auto pmem = getOperandPmemOfFormatType(filter, memory::format::oihw, type_conv_filter);
            if (pmem == nullptr) {
                pmem = insertReorder(&filter, filter.pmem, memory::format::oihw, type_conv_filter, filter.scale);
            }*/

            filter.shape = {channels, depth_multiplier, 1, filter_height, filter_width};
            //add to stub pmem, then can be pick up later
            addStubPmem(&filter, new memory({{filter.shape, type_conv_filter, format_filter_group},
                                            *cpu_engine}, pmem->get_data_handle()));
        } else {
            nnAssert(filter.shape.size() == 5);
        }
    }

    //in case reorder.
    //the following is trying to find best format for input and filter
    auto md_conv_input = memory::desc(input.shape, type_conv_input, memory::format::any);
    auto md_conv_filter = memory::desc(filter.shape, type_conv_filter, memory::format::any);
    auto md_conv_bias = memory::desc(bias.shape, type_conv_bias, memory::format::any);
    //out is nchw
    auto md_conv_output = memory::desc(output.shape, type_conv_output, memory::format::any);

    memory::dims strides = {stride_height, stride_width};
    memory::dims paddings_l = {padding_top, padding_left};
    memory::dims paddings_r = {padding_bottom, padding_right};

    //get conv primitive_desc, it includes the best format for input and filter.
    auto desc_conv = mkldnn::convolution_forward::desc(mkldnn::prop_kind::forward,
            mkldnn::convolution_direct, md_conv_input, md_conv_filter, md_conv_bias,
            md_conv_output, strides, paddings_l, paddings_r, mkldnn::padding_kind::zero);
    auto primitive_desc_conv =
            mkldnn::convolution_forward::primitive_desc(desc_conv, *cpu_engine);


    //reorder for input?
    auto conv_input_desc = primitive_desc_conv.src_primitive_desc().desc();
   /* auto pmem_conv_input = getOperandPmemOfDesc(input, conv_input_desc);
    if (pmem_conv_input == nullptr)
        pmem_conv_input = insertReorder(&input, input.pmem, conv_input_desc, input.scale);*/
    auto pmem_conv_input = insertReorderIfNeed(&input, conv_input_desc);

    //reoder for filter?
    auto conv_filter_desc = primitive_desc_conv.weights_primitive_desc().desc();
    /*auto pmem_conv_filter = getOperandPmemOfDesc(filter, conv_filter_desc);
    if (pmem_conv_filter == nullptr)
        pmem_conv_filter = insertReorder(&filter, filter.pmem, conv_filter_desc, filter.scale);*/
    memory * pmem_conv_filter;
    if (group) {
        pmem_conv_filter = getOperandPmemOfDesc(filter, conv_filter_desc);
        if (pmem_conv_filter == nullptr) {
            auto format_filter_group = memory::format::goihw;
            auto pmem_group_filter = getOperandPmemOfFormatType(filter, format_filter_group,
                                                            type_conv_filter);
            pmem_conv_filter = insertReorder(pmem_group_filter, conv_filter_desc, true, filter.scale, filter.zero);
        }
    } else {
        pmem_conv_filter = insertReorderIfNeed(&filter, conv_filter_desc);
    }

    //reoder for filter?
    auto conv_bias_desc = primitive_desc_conv.bias_primitive_desc().desc();
    /*auto pmem_conv_bias = getOperandPmemOfDesc(filter, conv_bias_desc);
    if (pmem_conv_bias == nullptr)
        pmem_conv_bias = insertReorder(&bias, bias.pmem, conv_bias_desc, bias.scale);*/
    auto pmem_conv_bias = insertReorderIfNeed(&bias, conv_bias_desc);

    output.pmem = new memory(primitive_desc_conv.dst_primitive_desc());

    mNet.push_back(mkldnn::convolution_forward(primitive_desc_conv,
                       *pmem_conv_input, *pmem_conv_filter, *pmem_conv_bias, *output.pmem));

    //TODO: combine relu with conv, conv_relu does not provides src/dst/bias/weights_primitive_get
    if (activation != FusedActivationFunc::NONE) {
        output.pmem =  insertActivation(output.pmem, activation);
    }
    //pass the format that NN think this opertion output format.
    finalizeOutput(&output, memory::format::nhwc);

    return true;
}


bool MklDnnPreparedModel::importOperationPool(const Operation& operation)
{
    const hidl_vec<uint32_t>& ins = operation.inputs;
    const hidl_vec<uint32_t>& outs = operation.outputs;
    const size_t in_counts = ins.size();

    VLOG(L1, "import %s has inputs %zu",
             operation.type == OperationType::AVERAGE_POOL_2D ? "AVGPOOL" : "MAXPOOL",
             in_counts);

    if (in_counts != 10 && in_counts != 7) {
        return false;
    }

    bool explicit_padding = (in_counts == 10);

    RunTimeOperandInfo& input = mOperands[ins[0]];
    RunTimeOperandInfo& output = mOperands[outs[0]];

    initializeInput(&input, memory::format::nhwc);

    int32_t batches = input.shape[0];
    int32_t channels = input.shape[1];
    int32_t input_height = input.shape[2];
    int32_t input_width = input.shape[3];


    int32_t padding_left, padding_right;
    int32_t padding_top, padding_bottom;
    int32_t stride_width, stride_height;
    int32_t filter_width, filter_height;
    FusedActivationFunc activation;

    if (explicit_padding) {
        padding_left     = getScalarData<int32_t>(mOperands[ins[1]]);
        padding_right    = getScalarData<int32_t>(mOperands[ins[2]]);
        padding_top      = getScalarData<int32_t>(mOperands[ins[3]]);
        padding_bottom   = getScalarData<int32_t>(mOperands[ins[4]]);
        stride_width     = getScalarData<int32_t>(mOperands[ins[5]]);
        stride_height    = getScalarData<int32_t>(mOperands[ins[6]]);
        filter_width     = getScalarData<int32_t>(mOperands[ins[7]]);
        filter_height    = getScalarData<int32_t>(mOperands[ins[8]]);
        activation       = getScalarData<FusedActivationFunc>(mOperands[ins[9]]);
    } else {
        int32_t padding_implicit = getScalarData<int32_t>(mOperands[ins[1]]);
        stride_width     = getScalarData<int32_t>(mOperands[ins[2]]);
        stride_height    = getScalarData<int32_t>(mOperands[ins[3]]);
        filter_width     = getScalarData<int32_t>(mOperands[ins[4]]);
        filter_height    = getScalarData<int32_t>(mOperands[ins[5]]);
        activation       = getScalarData<FusedActivationFunc>(mOperands[ins[6]]);

        calculateExplicitPadding(input_width, stride_width,
                                 filter_width, padding_implicit,
                                 &padding_left, &padding_right);
        calculateExplicitPadding(input_height, stride_height,
                                 filter_height, padding_implicit,
                                 &padding_top, &padding_bottom);
    }

    int32_t output_height = computeOutSize(input_height, filter_height, stride_height,
                                        padding_top, padding_bottom);
    int32_t output_width = computeOutSize(input_width, filter_width, stride_width,
                                       padding_left, padding_right);

    VLOG(L2, "batches %d, channels %d, input_height: %d, input_width %d",
             batches, channels, input_height, input_width);
    VLOG(L2, "filter_height: %d, filter_width %d", filter_height, filter_width);
    VLOG(L2, "padding: top: %d, left %d, bottom %d, right %d", padding_top, padding_left, padding_bottom, padding_right);
    VLOG(L2, "stride: height %d, width %d", stride_height, stride_width);

    //get output shape, mkldnn define shape as nchw
    output.shape = {batches, channels, output_height, output_width};

    //pooling acccept only nchw
    auto type_pool_input = getOperandNeedType(input);
    /*auto pmem_pool_input = getOperandPmemOfFormatType(input, memory::format::nchw,
                                                      type_pool_input);
    if (pmem_pool_input == nullptr) {
        pmem_pool_input = insertReorder(&input, input.pmem, memory::format::nchw,
                                        type_pool_input, input.scale);
    }*/
    auto pmem_pool_input = insertReorderIfNeed(&input, memory::format::nchw, type_pool_input);

    //output has same type as input
    auto md_pool_output = memory::desc(output.shape, type_pool_input, memory::format::any);
    auto alg = mkldnn::algorithm::pooling_max;
    if (operation.type == OperationType::AVERAGE_POOL_2D)
        alg = mkldnn::algorithm::pooling_avg;

    memory::dims strides = {stride_height, stride_width};
    memory::dims paddings_l = {padding_top, padding_left};
    memory::dims paddings_r = {padding_bottom, padding_right};
    memory::dims kernel = {filter_height, filter_width};

    auto desc_pool = mkldnn::pooling_forward::desc(mkldnn::prop_kind::forward_inference, alg,
            pmem_pool_input->get_primitive_desc().desc(), md_pool_output, strides,
            kernel, paddings_l, paddings_r, mkldnn::padding_kind::zero);

    auto primitive_desc_pool =
                mkldnn::pooling_forward::primitive_desc(desc_pool, *cpu_engine);

    output.pmem = new memory(primitive_desc_pool.dst_primitive_desc());
    /* create pooling primitive an add it to net */
    mNet.push_back(mkldnn::pooling_forward(primitive_desc_pool, *pmem_pool_input,
                     *output.pmem));

    if (activation != FusedActivationFunc::NONE) {
        output.pmem = insertActivation(output.pmem, activation);
    }

    finalizeOutput(&output, memory::format::nhwc);
    return true;
}

bool MklDnnPreparedModel::importOperationActivation(const Operation& operation)
{
    const hidl_vec<uint32_t>& ins = operation.inputs;
    const hidl_vec<uint32_t>& outs = operation.outputs;
    const size_t in_counts = ins.size();

    VLOG(L1, "import ACTIVATION type %d, has inputs %zu", operation.type, in_counts);

    if (in_counts != 1) {
        return false;
    }

    mkldnn::algorithm alg;
    float alpha = 0, beta = 0;
    switch (operation.type) {
        case OperationType::RELU:
            alg = mkldnn::algorithm::eltwise_relu;
            break;
        case OperationType::RELU6:
            alg = mkldnn::algorithm::eltwise_bounded_relu;
            alpha = 6;
            break;
        case OperationType::LOGISTIC:
            alg = mkldnn::algorithm::eltwise_logistic;
            break;
        case OperationType::TANH:
            alg = mkldnn::algorithm::eltwise_tanh;
            break;
        default:
            nnAssert(false);
    }

    RunTimeOperandInfo& input = mOperands[ins[0]];
    RunTimeOperandInfo& output = mOperands[outs[0]];

    initializeInput(&input, memory::format::nhwc);

    auto type_activation_input = getOperandNeedType(input);
    /*auto pmem_relu_input = getOperandPmemOfFormatType(input, input.format, type_relu_input);
    if (pmem_relu_input == nullptr) {
        pmem_relu_input = insertReorder(&input, input.pmem, input.format, type_relu_input, input.scale);
    }*/
    auto pmem_activation_input = insertReorderIfNeed(&input, input.format, type_activation_input);

    output.shape = input.shape;

    //get output shape, mkldnn define shape as nchw
    //output has same type as input
    output.pmem = new memory({{output.shape, type_activation_input, input.format},
                                      *cpu_engine});

    /* create relu primitive and add it to net */
    auto desc_activation = mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward,
                                   alg, pmem_activation_input->get_primitive_desc().desc(), alpha, beta);
    auto primitive_desc_activation = mkldnn::eltwise_forward::primitive_desc(desc_activation,
                        *cpu_engine);

    mNet.push_back(mkldnn::eltwise_forward(primitive_desc_activation, *pmem_activation_input,
                        *output.pmem));

    //output format same as input, and do not need reorder.
    finalizeOutput(&output, input.format);
    return true;
}

bool MklDnnPreparedModel::importOperationConcat(const Operation& operation)
{
    const hidl_vec<uint32_t>& ins = operation.inputs;
    const hidl_vec<uint32_t>& outs = operation.outputs;
    const size_t in_counts = ins.size();

    VLOG(L1, "import CONCAT has inputs %zu", in_counts);

    nnAssert(in_counts >= 2);

    uint32_t axis = getScalarData<uint32_t>(mOperands[ins[in_counts - 1]]);
    RunTimeOperandInfo& input0 = mOperands[ins[0]];
    int dims_size = input0.dims.size();

    memory::format format_concat;
    memory::format format_input;
    memory::format format_output;
    memory::dims shape_output;
    switch(dims_size) {
        case 1:
            format_concat = memory::format::x;
            format_input = memory::format::x;
            format_output = memory::format::x;
            shape_output = {0};
            break;
        case 2:
            format_concat = memory::format::nc;
            format_input = memory::format::nc;
            format_output = memory::format::nc;
            shape_output = {0, 0};
            break;
        case 4:
            format_concat = memory::format::nchw;
            format_input = memory::format::nchw;
            format_output = memory::format::nchw;
            shape_output = {0, 0, 0, 0};
            axis = (axis == 0) ? 0 : (axis % 3) + 1;
            break;
        default:
            ALOGE("unsupported dims size %d", dims_size);
            return false;
    }

    auto type_concat_input = getOperandNeedType(input0);
    std::vector<memory::primitive_desc> primitive_desc_inputs;
    std::vector<primitive::at> primitive_inputs;
    for (uint32_t i = 0; i < in_counts - 1; i++) {
        RunTimeOperandInfo& input = mOperands[ins[i]];
        initializeInput(&input, format_input);

        //based on nchw
        /*auto pmem_input = getOperandPmemOfFormatType(input, format_concat, type_concat_input);
        if (pmem_input == nullptr) {
            pmem_input = insertReorder(&input, input.pmem, format_concat, type_concat_input, input.scale);
        }*/
        auto pmem_input = insertReorderIfNeed(&input, format_concat, type_concat_input);
        primitive_inputs.push_back(*pmem_input);
        primitive_desc_inputs.push_back(pmem_input->get_primitive_desc());

        //axis is already nchw, then use input.shape
        for (uint32_t d = 0; d < input.shape.size(); d++) {
            if (d == axis) {
                shape_output[d] += input.shape[d];
            } else {
                if (shape_output[d] == 0) {
                    shape_output[d] = input.shape[d];
                } else {
                    nnAssert(input.shape[d] == shape_output[d]);
                }
            }
        }
    }

    RunTimeOperandInfo& output = mOperands[outs[0]];
    output.shape = shape_output;

    auto desc_output = memory::desc(output.shape, type_concat_input, format_concat);
    auto primitive_desc_concat = mkldnn::concat::primitive_desc(desc_output,
                                          static_cast<int>(axis),
                                          primitive_desc_inputs);
    output.pmem = new memory(primitive_desc_concat.dst_primitive_desc());

    mNet.push_back(mkldnn::concat(primitive_desc_concat, primitive_inputs, *output.pmem));

    finalizeOutput(&output, format_output);
    return true;
}

bool MklDnnPreparedModel::importOperationSoftmax(const Operation& operation)
{
    const hidl_vec<uint32_t>& ins = operation.inputs;
    const hidl_vec<uint32_t>& outs = operation.outputs;
    const size_t in_counts = ins.size();

    VLOG(L1, "import SOFTMAX has inputs %zu", in_counts);

    nnAssert(in_counts == 2);

    RunTimeOperandInfo& input = mOperands[ins[0]];
    float beta = getScalarData<float>(mOperands[ins[1]]);

    //do not support beta in mkldnn
    nnAssert(beta == 1.0f);

    int dims_size = input.dims.size();
    memory::format format_softmax;
    memory::format format_input;
    memory::format format_output;
    memory::dims shape_output;
    switch(dims_size) {
        case 2:
            format_softmax = memory::format::nc;
            format_input = memory::format::nc;
            format_output = memory::format::nc;
            break;
        case 4:
            format_softmax = memory::format::nchw;
            format_input = memory::format::nhwc;
            format_output = memory::format::nhwc;
            break;
        default:
            LOG(ERROR) << "unsupported dims size";
            return false;
    }

    auto type_softmax_input = getOperandNeedType(input);

    initializeInput(&input, format_input);
    /*auto pmem_softmax_input = getOperandPmemOfFormatType(input, format_softmax, type_softmax_input);
    if (pmem_softmax_input == nullptr) {
        pmem_softmax_input = insertReorder(&input, input.pmem, format_softmax, type_softmax_input, input.scale);
    }*/
    auto pmem_softmax_input = insertReorderIfNeed(&input, format_softmax, type_softmax_input);

    RunTimeOperandInfo& output = mOperands[outs[0]];
    output.shape = input.shape;
    output.pmem = new memory({{output.shape, type_softmax_input, format_output}, *cpu_engine});

    auto desc_softmax = mkldnn::softmax_forward::desc(mkldnn::prop_kind::forward_inference,
                                  pmem_softmax_input->get_primitive_desc().desc(), dims_size - 1);
    auto primitive_desc_softmax = mkldnn::softmax_forward::primitive_desc(desc_softmax, *cpu_engine);
    mNet.push_back(mkldnn::softmax_forward(primitive_desc_softmax,
                                  *pmem_softmax_input, *output.pmem));

    finalizeOutput(&output, format_output);
    return true;
}

bool MklDnnPreparedModel::importOperationLRN(const Operation& operation)
{
    const hidl_vec<uint32_t>& ins = operation.inputs;
    const hidl_vec<uint32_t>& outs = operation.outputs;
    const size_t in_counts = ins.size();

    VLOG(L1, "import LRN has inputs %zu", in_counts);

    nnAssert(in_counts == 5);

    RunTimeOperandInfo& input = mOperands[ins[0]];
    initializeInput(&input, memory::format::nhwc);
    //mkldnn accept nchw;
    auto format_lrn = memory::format::nchw;
    auto format_output = memory::format::nchw;
    auto type_lrn_input = getOperandNeedType(input);
    /*auto pmem_lrn_input = getOperandPmemOfFormatType(input, softmax_format);
    if (pmem_lrn_input == nullptr) {
        pmem_lrn_input = insertReorder(&input, input.pmem, softmax_format);
    }*/
    auto pmem_lrn_input = insertReorderIfNeed(&input, format_lrn, type_lrn_input);

    int32_t radius = getScalarData<int32_t>(mOperands[ins[1]]);
    float bias = getScalarData<float>(mOperands[ins[2]]);
    float alpha = getScalarData<float>(mOperands[ins[3]]);
    float beta = getScalarData<float>(mOperands[ins[4]]);

    RunTimeOperandInfo& output = mOperands[outs[0]];
    output.shape = input.shape;
    output.pmem = new memory({{output.shape, type_lrn_input, format_output}, *cpu_engine});

    //NN pass the depth as (d - depth, d + depth)
    radius = radius * 2 + 1;
    //mkldnn take sum = k + alpha * sqr_sum/radius; while NN does not have /radius
    alpha = alpha * radius;

    auto desc_lrn = mkldnn::lrn_forward::desc(mkldnn::prop_kind::forward_scoring,
                                              mkldnn::algorithm::lrn_across_channels,
                                              pmem_lrn_input->get_primitive_desc().desc(),
                                              radius, alpha, beta, bias);
    auto primitive_desc_lrn = mkldnn::lrn_forward::primitive_desc(desc_lrn, *cpu_engine);
    mNet.push_back(mkldnn::lrn_forward(primitive_desc_lrn, *pmem_lrn_input, *output.pmem));

    finalizeOutput(&output, memory::format::nhwc);
    return true;
}

bool MklDnnPreparedModel::importOperationFC(const Operation& operation)
{
    const hidl_vec<uint32_t>& ins = operation.inputs;
    const hidl_vec<uint32_t>& outs = operation.outputs;
    const size_t in_counts = ins.size();

    VLOG(L1, "import FULLCONNECT has inputs %zu", in_counts);

    nnAssert(in_counts == 4);

    RunTimeOperandInfo& input = mOperands[ins[0]];
    RunTimeOperandInfo& weights = mOperands[ins[1]];
    RunTimeOperandInfo& bias = mOperands[ins[2]];

    auto activation = getScalarData<FusedActivationFunc>(mOperands[ins[3]]);

    initializeInput(&input, memory::format::nc);
    initializeInput(&weights, memory::format::nc);
    initializeInput(&bias, memory::format::x);

    //input is [batch_size, input_size], weights is [num_unit, input_size]
    nnAssert(input.shape[1] == weights.shape[1]);

    auto type_fc_input = getOperandNeedType(input);
    auto type_fc_weights = getOperandNeedType(weights);
    auto type_fc_bias = getOperandNeedType(bias);

    /*auto pmem_fc_input = getOperandPmemOfFormatType(input, input.format, type_fc_input);
    if (pmem_fc_input == nullptr) {
        pmem_fc_input = insertReorder(&input, input.pmem, input.format, type_fc_input, input.scale);
    }*/
    auto getCompatiblePmem = [&](RunTimeOperandInfo* operand, memory::format format, memory::data_type type) -> memory * {
        auto src_pmem = operand->pmem;
        auto desc_src = src_pmem->get_primitive_desc().desc();
        //we only compare format and type
        auto format_src = static_cast<memory::format>(desc_src.data.format);
        auto type_src = static_cast<memory::data_type>(desc_src.data.data_type);
        bool can_shrink = true;
        memory* pmem_fc;
        memory::dims shape;
        //for nc
        shape.resize(2);
        if (format_src == format && type_src == type) {
            return src_pmem;
        } else if (format_src != format && type_src == type) {
            for (int i = 0; i < desc_src.data.ndims; i++) {
                if (i < shape.size()) {
                    shape[i] = desc_src.data.dims[i];
                } else {
                    if (desc_src.data.dims[i] != 1) {
                        can_shrink = false;
                        break;
                    }
                }
            }
        } else {
            can_shrink = false;
        }
        if (can_shrink) {
            void *buf = src_pmem->get_data_handle();
            pmem_fc = new memory({{shape, type, format}, *cpu_engine}, buf);
        } else {
            pmem_fc = insertReorderIfNeed(operand, format, type);
        }
        return pmem_fc;
    };

    auto pmem_fc_input = getCompatiblePmem(&input, memory::format::nc, type_fc_input);
    auto pmem_fc_weights = getCompatiblePmem(&weights, memory::format::nc, type_fc_weights);
    auto pmem_fc_bias = getCompatiblePmem(&bias, memory::format::x, type_fc_bias);

    RunTimeOperandInfo& output = mOperands[outs[0]];
    output.shape = {input.shape[0], weights.shape[0]};
    //output has same type as input
    output.pmem = new memory({{output.shape, type_fc_input, memory::format::nc}, *cpu_engine});

    auto desc_fc = mkldnn::inner_product_forward::desc(mkldnn::prop_kind::forward,
                               pmem_fc_input->get_primitive_desc().desc(),
                               pmem_fc_weights->get_primitive_desc().desc(),
                               pmem_fc_bias->get_primitive_desc().desc(),
                               output.pmem->get_primitive_desc().desc());
    auto primitive_desc_fc = mkldnn::inner_product_forward::primitive_desc(desc_fc, *cpu_engine);
    mNet.push_back(mkldnn::inner_product_forward(primitive_desc_fc, *pmem_fc_input,
                                                       *pmem_fc_weights, *pmem_fc_bias, *output.pmem));

    if (activation != FusedActivationFunc::NONE) {
        output.pmem = insertActivation(output.pmem, activation);
    }

    finalizeOutput(&output, memory::format::nc);
    return true;
}

bool MklDnnPreparedModel::importOperationAdd(const Operation& operation)
{
    const hidl_vec<uint32_t>& ins = operation.inputs;
    const hidl_vec<uint32_t>& outs = operation.outputs;
    const size_t in_counts = ins.size();

    VLOG(L1, "import ADD has inputs %zu", in_counts);

    nnAssert(in_counts == 3);

    auto activation = getScalarData<FusedActivationFunc>(mOperands[ins[in_counts - 1]]);
    RunTimeOperandInfo& input0 = mOperands[ins[0]];
    memory::format format_input;

    //TODO: workaround 3-D
    int dims_size = input0.dims.size();
    switch(dims_size) {
        case 2:
            format_input = memory::format::nc;
            break;
        case 4:
            format_input = memory::format::nchw;
            break;
        case 1:
            format_input = memory::format::x;
            break;
        default:
            ALOGE("unsupported dims size %d", dims_size);
            return false;
    }

    std::vector<float> scales = {1, 1, 1, 1};

    std::vector<mkldnn::memory::primitive_desc> md_inputs;
    std::vector<primitive::at> inputs;
    for (uint32_t i = 0; i < in_counts - 1; i++) {
        RunTimeOperandInfo& input = mOperands[ins[i]];
        initializeInput(&input, format_input);
        auto type_input = getOperandNeedType(input);
        auto pmem_input = insertReorderIfNeed(&input, format_input, type_input);
        md_inputs.push_back(pmem_input->get_primitive_desc());
        inputs.push_back(*pmem_input);
    }

    auto pd_add = mkldnn::sum::primitive_desc(scales, md_inputs);

    RunTimeOperandInfo& output = mOperands[outs[0]];
    output.pmem = new memory(pd_add.dst_primitive_desc());
    //output shape same as input
    output.shape = input0.shape;

    mNet.push_back(mkldnn::sum(pd_add, inputs, *output.pmem));

    if (activation != FusedActivationFunc::NONE) {
        output.pmem = insertActivation(output.pmem, activation);
    }

    finalizeOutput(&output, format_input);

    return true;
}

// TODO doublecheck
bool MklDnnPreparedModel::validateRequest(const Request& request, const Model& model)
{
    auto validRequestArguments = [&](const hidl_vec<RequestArgument>& arguments,
                                     const hidl_vec<uint32_t>& operandIndexes,
                                     const hidl_vec<Operand>& operands, size_t poolCount,
                                     const char* type) -> bool {
        const size_t argumentCount = arguments.size();
        if (argumentCount != operandIndexes.size()) {
            ALOGE("Request specifies %zu, but %s has the model as %zu", argumentCount, type,
                  operandIndexes.size());
            return false;
        }
        for (size_t argumentIndex = 0; argumentIndex < argumentCount; argumentIndex++) {
            const RequestArgument& argument = arguments[argumentIndex];
            const uint32_t operandIndex = operandIndexes[argumentIndex];
            const Operand& operand = operands[operandIndex];
            if (argument.hasNoValue) {
                if (argument.location.poolIndex != 0 ||
                    argument.location.offset != 0 ||
                    argument.location.length != 0 ||
                    argument.dimensions.size() != 0) {
                    ALOGE("Request %s: %zu has no value yet has details.", type, argumentIndex);
                    return false;
                }
            }
            if (argument.location.poolIndex >= poolCount) {
                ALOGE("Request %s: %zu has an invalid poolIndex %d", type, argumentIndex,
                      argument.location.poolIndex);
                return false;
            }
            // TODO: Validate that we are within the pool.
            uint32_t rank = argument.dimensions.size();
            if (rank > 0) {
                if (rank != operand.dimensions.size()) {
                    ALOGE("Request %s: %zu  has number of dimensions (%d) different than the model's (%zu)",
                          type, argumentIndex, rank, operand.dimensions.size());
                    return false;
                }
                for (size_t i = 0; i < rank; i++) {
                    if (argument.dimensions[i] != operand.dimensions[i] &&
                        operand.dimensions[i] != 0) {
                        ALOGE("Request %s: %zu has dimension %zu of %d different than the model's %d",
                              type, argumentIndex, i, operand.dimensions[i], operand.dimensions[i]);
                        return false;
                    }
                    if (argument.dimensions[i] == 0) {
                        ALOGE("Request %s: %zu has dimension %zu of zero", type, argumentIndex, i);
                        return false;
                    }
                }
            }
        }
        return true;
    };

    const size_t pool_counts = request.pools.size();
    return (validRequestArguments(request.inputs, model.inputIndexes, model.operands, pool_counts,
                                  "input") &&
            validRequestArguments(request.outputs, model.outputIndexes, model.operands, pool_counts,
                                  "output"));
}

bool MklDnnPreparedModel::validModel(const Model& model)
{
    auto validOperandIndexes = [&](const hidl_vec<uint32_t> indexes, size_t operandCount) -> bool {
        for (uint32_t i : indexes) {
            if (i >= operandCount) {
                ALOGE("Index out of range %d/%zu",i, operandCount);
                return false;
            }
        }
        return true;
    };

    const size_t operand_counts = model.operands.size();

    if (!validOperandIndexes(model.inputIndexes, operand_counts) ||
        !validOperandIndexes(model.outputIndexes, operand_counts)) {
        ALOGE("model inputs/outputs index invalid");
        return false;
    }

    for (size_t i = 0; i < operand_counts; i++) {
        const Operand& operand = model.operands[i];
        if (operand.type != OperandType::TENSOR_FLOAT32 &&
            operand.type != OperandType::FLOAT32 &&
            operand.type != OperandType::INT32 &&
            operand.type != OperandType::UINT32 &&
            operand.type != OperandType::TENSOR_INT32 &&
            operand.type != OperandType::TENSOR_QUANT8_ASYMM) {
            ALOGE("wrong operand type %d", operand.type);
            return false;
         }

        switch (operand.lifetime) {
            case OperandLifeTime::MODEL_INPUT:
            case OperandLifeTime::MODEL_OUTPUT:
            case OperandLifeTime::NO_VALUE:
            case OperandLifeTime::TEMPORARY_VARIABLE:
                if (operand.location.offset != 0 || operand.location.length != 0) {
                    ALOGE("Unexpected offset %d, or length %d",
                          operand.location.offset, operand.location.length);
                    return false;
                }
                break;
            case OperandLifeTime::CONSTANT_COPY:
                if (operand.location.offset + operand.location.length > model.operandValues.size()) {
                    ALOGE("OperandValue location out of range.  Starts at %d length %d. max %zu", 
                          operand.location.offset, operand.location.length, model.operandValues.size());
                    return false;
                }
                break;
            case OperandLifeTime::CONSTANT_REFERENCE:
            {
                auto pool_counts = model.pools.size();
                if (operand.location.poolIndex >= pool_counts) {
                    ALOGE("Invalid poolIndex %d/%lu", operand.location.poolIndex, pool_counts);
                    return false;
                }
                break;
            }
            default:
                return false;
        }
    }

    for (const auto& operation : model.operations) {
        //TANH is last one
        if (static_cast<uint32_t>(operation.type) >
            static_cast<uint32_t>(OperationType::TANH)) {
            return false;
        }
    }

    return true;
}

bool MklDnnPreparedModel::initializeRunTimeOperandInfo() {
    //initialize runtime operand info from model.
    const size_t count = mModel.operands.size();
    mOperands.resize(count);
    // Start by setting the runtime info to what's in the model.
    for (size_t i = 0; i < count; i++) {
        const Operand& from = mModel.operands[i];
        RunTimeOperandInfo& to = mOperands[i];
        to.dims.resize(from.dimensions.size());
        for (size_t j = 0; j < from.dimensions.size(); j++) {
            to.dims[j] = from.dimensions[j];
        }
        to.scale = from.scale;
        nnAssert(from.zeroPoint == 0);
        switch(from.type) {
            case OperandType::TENSOR_FLOAT32:
            case OperandType::FLOAT32:
                //nnAssert(to.scale == 0);
                to.scale = 0;
                to.type = memory::data_type::f32;
                break;
            case OperandType::INT32:
            case OperandType::UINT32:
                nnAssert(to.scale == 0);
            case OperandType::TENSOR_INT32:
                to.type = memory::data_type::s32;
                break;
            case OperandType::TENSOR_QUANT8_ASYMM:
                nnAssert(to.scale != 0);
                to.type = memory::data_type::u8;
                break;
            default:
                ALOGE("wrong operand type %d", from.type);;
                return false;
        }
        to.pmem = nullptr;
        to.stub_pmems = {};
        to.format = memory::format::format_undef,
        to.length = from.location.length;
        to.lifetime = from.lifetime;
        switch (from.lifetime) {
            case OperandLifeTime::TEMPORARY_VARIABLE:
                to.buffer = nullptr;
                to.length = sizeOfData(to.type, to.dims);
                to.numberOfUsesLeft = from.numberOfConsumers;
                break;
            case OperandLifeTime::CONSTANT_COPY:
                to.buffer = const_cast<uint8_t*>(&mModel.operandValues[from.location.offset]);
                to.numberOfUsesLeft = 0;
                break;
            case OperandLifeTime::CONSTANT_REFERENCE:
            {
                auto poolIndex = from.location.poolIndex;
                nnAssert(poolIndex < mPoolInfos.size());
                auto& r = mPoolInfos[poolIndex];
                to.buffer = r.buffer + from.location.offset;
                to.numberOfUsesLeft = 0;
                break;
            }
            case OperandLifeTime::MODEL_INPUT:
            case OperandLifeTime::MODEL_OUTPUT:
            case OperandLifeTime::NO_VALUE:
                to.buffer = nullptr;
                to.length = sizeOfData(to.type, to.dims);
                to.numberOfUsesLeft = 0;
                break;
            default:
                return false;
                break;
        }
    }
    return true;
}

bool MklDnnPreparedModel::initialize()
{
    bool success = false;

    //Check operation supoorted or not, user may not call getOpertionSupported()
    for (const auto& operation : mModel.operations) {
        success = isOperationSupported(operation, mModel);
        if (!success) {
            ALOGE("get unsupported operation in initialize()");
            return false;
        }
    }

    success = setRunTimePoolInfosFromHidlMemories(&mPoolInfos, mModel.pools);
    if (!success) {
        ALOGE("setRunTimePoolInfosFromHidlMemories failed.");
        return false;
    }

    cpu_engine = new engine(engine::cpu, 0);

    success = initializeRunTimeOperandInfo();
    if (!success) {
        ALOGE("initializeRunTimeOperandInfo failed.");
        return false;
    }

    for (const auto& operation : mModel.operations) {
        VLOG(L1, "get operation %d ready to import", operation.type);
       switch (operation.type) {
            case OperationType::CONV_2D:
            case OperationType::DEPTHWISE_CONV_2D:
                success = importOperationConv2D(operation);
                break;
            case OperationType::MAX_POOL_2D:
            case OperationType::AVERAGE_POOL_2D:
                success = importOperationPool(operation);
                break;
            case OperationType::RELU:
            case OperationType::RELU6:
            case OperationType::LOGISTIC:
            case OperationType::TANH:
                success = importOperationActivation(operation);
                break;
            case OperationType::CONCATENATION:
                success = importOperationConcat(operation);
                break;
            case OperationType::SOFTMAX:
                success = importOperationSoftmax(operation);
                break;
            case OperationType::LOCAL_RESPONSE_NORMALIZATION:
                success = importOperationLRN(operation);
                break;
            case OperationType::FULLY_CONNECTED:
                success = importOperationFC(operation);
                break;
            case OperationType::ADD:
                success = importOperationAdd(operation);
                break;
            default:
                ALOGE("unsupported operation %d", operation.type);
                return false;
        }
        if (success == false) {
                ALOGE("failed to import operation %d", operation.type);
                return false;
        }
        VLOG(L1, "import %d success", operation.type);
    }

    return true;
}

void MklDnnPreparedModel::deinitialize()
{
    VLOG(L1,  "deinitialize");
    for (const auto& operand : mOperands) {
        for (const auto& pmem : operand.stub_pmems) {
            VLOG(L1, "free stub pmems %p of operand %p", pmem, &operand);
            delete pmem;
        }
        VLOG(L1, "free pmems %p of operand %p", operand.pmem, &operand);
        if (operand.pmem)
            delete operand.pmem;
    }
    VLOG(L1, "free cpu engine");
    if (cpu_engine)
        delete cpu_engine;
}

#ifdef MKLDNN_DEBUG
template <typename T>
void printBuffer(int level, T* buf, int num, int items, const char* format)
{
    char str[1024];
    int start = 0;
    int n = 0;
    while (n < num) {
        int offset = 0;
        n = (n + items) > num ? num : n + items;
        offset = sprintf(str, "[%d->%d]:\t", start, n);
        for (int i = start; i < n; i++) {
            offset += sprintf(str + offset, format, buf[i]);
        }
        start = n;
        VLOG(level, "%s", str);
    }
}

void printOperandPmem(int level, const memory *pmem, int limit = 0)
{
    auto desc = pmem->get_primitive_desc().desc();
    auto type = static_cast<mkldnn::memory::data_type>(desc.data.data_type);
    int size = 1;
    for (int i = 0; i < desc.data.ndims; i++)
        size *= desc.data.dims[i];
    if (limit > 0 && limit < size)
        size = limit;

    VLOG(level, "pmem %p: format %d, type %d", pmem, desc.data.format, type);
    if (type == memory::data_type::f32) {
        float *buf = static_cast<float *>(pmem->get_data_handle());
        printBuffer<float>(level, buf, size, 10, "%f\t");
    } else if (type == memory::data_type::s32) {
        int32_t *buf = static_cast<int32_t *>(pmem->get_data_handle());
        printBuffer<int32_t>(level, buf, size, 10, "%d\t");
    } else if (type == memory::data_type::u8) {
        uint8_t *buf = static_cast<uint8_t *>(pmem->get_data_handle());
        printBuffer<uint8_t>(level, buf, size, 10, "%d\t");
    } else {
        VLOG(level, "Do not support type %d", type);
    }
}
#endif

void MklDnnPreparedModel::asyncExecute(const Request& request,
                                       const sp<IExecutionCallback>& callback)
{
    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!setRunTimePoolInfosFromHidlMemories(&requestPoolInfos, request.pools)) {
        callback->notify(ErrorStatus::GENERAL_FAILURE);
        return;
    }

    auto copyData = [this, &requestPoolInfos](const std::vector<uint32_t>& indexes,
                       const hidl_vec<RequestArgument>& arguments, bool copyFromRequest) {
        //do memcpy for input data
        for (size_t i = 0; i < indexes.size(); i++) {
            RunTimeOperandInfo& operand = mOperands[indexes[i]];
            const RequestArgument& arg = arguments[i];
            auto poolIndex = arg.location.poolIndex;
            nnAssert(poolIndex < requestPoolInfos.size());
            auto& r = requestPoolInfos[poolIndex];
            if (copyFromRequest)
                memcpy(operand.buffer, r.buffer + arg.location.offset, operand.length);
            else
                memcpy(r.buffer + arg.location.offset, operand.buffer, operand.length);
        }
    };

    VLOG(L1, "copy request inputs to model inputs");

    copyData(mModel.inputIndexes, request.inputs, true);

    VLOG(L1, "Run");
    //run
    mkldnn::stream(mkldnn::stream::kind::eager).submit(mNet).wait();

    VLOG(L1, "copy model output to request output");

    copyData(mModel.outputIndexes, request.outputs, false);

    VLOG(L1, "update shared memories");
    for (auto runtimeInfo : requestPoolInfos) {
        runtimeInfo.update();
    }

#ifdef MKLDNN_DEBUG
    {
        VLOG(L2, "Model output0 are:");
        const RunTimeOperandInfo& output = mOperands[mModel.outputIndexes[0]];
        printOperandPmem(L2, output.pmem);
        VLOG(L4, "Model input0 are:");
        const RunTimeOperandInfo& input = mOperands[mModel.inputIndexes[0]];
        printOperandPmem(L4, input.pmem, 100);
        for(const auto& op : mModel.operations) {
            const auto& o = mOperands[op.outputs[0]];
            VLOG(L4, "Operation %d has output 0(lifetime %d) are:", op.type, o.lifetime);
            printOperandPmem(L4, o.pmem, 100);
        }
    }
#endif

    Return<void> returned = callback->notify(ErrorStatus::NONE);
    if (!returned.isOk()) {
        ALOGE("hidl callback failed to return properly: %s", returned.description().c_str());
    }
}

Return<ErrorStatus> MklDnnPreparedModel::execute(const Request& request,
                                                 const sp<IExecutionCallback>& callback)
{
    VLOG(L1, "Begin to execute");

    if (mNet.size() == 0) {
        ALOGE("No primitive to execute");
        callback->notify(ErrorStatus::INVALID_ARGUMENT);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    if (callback.get() == nullptr) {
        ALOGE("invalid callback passed to execute");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    if (validateRequest(request, mModel) == false) {
        callback->notify(ErrorStatus::INVALID_ARGUMENT);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // This thread is intentionally detached because the sample driver service
    // is expected to live forever.
    std::thread([this, request, callback]{ asyncExecute(request, callback); }).detach();

    VLOG(L1, "Start execute thread done");
    return ErrorStatus::NONE;
}

template <typename T>
T getOperandConstVal(const Model& model, const Operand& operand)
{
    const T* data = reinterpret_cast<const T *>(&model.operandValues[operand.location.offset]);
    return data[0];
}

bool MklDnnPreparedModel::isOperationSupported(const Operation& operation, const Model& model)
{
    VLOG(L1, "Check operation %d", operation.type);

    #define VLOG_CHECKFAIL(fail)  VLOG(L1, "Check failed: %s", fail)

#ifdef DISABLE_ALL_QUANT
    for (auto i : operation.inputs) {
        const auto input = model.operands[i];
        if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
            VLOG_CHECKFAIL("input quant");
            return false;
        }
    }
    for (auto i : operation.outputs) {
        const auto output = model.operands[i];
        if (output.type == OperandType::TENSOR_QUANT8_ASYMM) {
            VLOG_CHECKFAIL("output quant");
            return false;
        }
    }
#else
    for (auto i : operation.inputs) {
        const auto input = model.operands[i];
        if (input.type == OperandType::TENSOR_QUANT8_ASYMM &&
            input.zeroPoint != 0) {
            VLOG_CHECKFAIL("input quant");
            return false;
        }
    }
    for (auto i : operation.outputs) {
        const auto output = model.operands[i];
        if (output.type == OperandType::TENSOR_QUANT8_ASYMM &&
            output.zeroPoint != 0) {
            VLOG_CHECKFAIL("output quant");
            return false;
        }
    }
#endif

    const auto input0 = model.operands[operation.inputs[0]];
    auto activationPass = [&model](const Operand& input) -> bool {
        const FusedActivationFunc activation = getOperandConstVal<FusedActivationFunc>(model, input);
        if (activation == FusedActivationFunc::RELU1) {
            VLOG_CHECKFAIL("relu1 used");
            return false;
        }
        return true;
    };

    const auto& inputn = model.operands[operation.inputs[operation.inputs.size() - 1]];
    switch(operation.type) {
        case OperationType::DEPTHWISE_CONV_2D:
        {
            const auto& input1 = model.operands[operation.inputs[1]];
            //channels_out must be channels * depth_mul
            if ((input1.dimensions[3] % input0.dimensions[3]) != 0) {
                VLOG_CHECKFAIL("dims not in group");
                return false;
            }
            if (activationPass(inputn) == false) {
                return false;
            }
            break;
        }
        case OperationType::SOFTMAX:
        {
            const auto& input1 = model.operands[operation.inputs[1]];
            float beta = getOperandConstVal<float>(model, input1);
            //beta need = 1.0f
            if (beta != 1.0f) {
                VLOG_CHECKFAIL("beta not 1.0f");
                return false;
            }
            break;
        }
        case OperationType::CONV_2D:
        {
            const auto& input1 = model.operands[operation.inputs[1]];
            //filter in == channel
            if (input0.dimensions[3] != input1.dimensions[3]) {
                VLOG_CHECKFAIL("filter in not equals channel");
                return false;
            }
            //continue to check actication.
        }
        case OperationType::AVERAGE_POOL_2D:
        case OperationType::MAX_POOL_2D:
        case OperationType::FULLY_CONNECTED:
        {
            if (activationPass(inputn) == false) {
                return false;
            }
            break;
        }
        case OperationType::ADD:
        {
            const auto& input1 = model.operands[operation.inputs[1]];
            if (input0.dimensions != input1.dimensions) {
                VLOG_CHECKFAIL("dims not match");
                return false;
            }

            if (activationPass(inputn) == false) {
                return false;
            }
            break;
        }
        case OperationType::RELU:
        case OperationType::RELU6:
        case OperationType::LOGISTIC:
        case OperationType::TANH:
        case OperationType::LOCAL_RESPONSE_NORMALIZATION:
        case OperationType::CONCATENATION:
             break;
        default:
           VLOG(L1, "unsupport opration %d", operation.type);
           return false;
    }
    VLOG(L1, "Operation %d supported", operation.type);

    return true;
}

}  // namespace mkldnn_driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
