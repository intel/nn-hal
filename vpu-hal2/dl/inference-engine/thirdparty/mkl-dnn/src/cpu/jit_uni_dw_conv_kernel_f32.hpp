/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef JIT_UNI_CONV_KERNEL_F32_HPP
#define JIT_UNI_CONV_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

template <cpu_isa_t isa>
struct jit_uni_dw_conv_fwd_kernel_f32: public jit_generator {
    jit_uni_dw_conv_fwd_kernel_f32(jit_conv_conf_t ajcp): jcp(ajcp)
    {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }


    static bool post_ops_ok(jit_conv_conf_t &jcp,
                            const primitive_attr_t &attr);
    static status_t init_conf(jit_conv_conf_t &jcp,
                              const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
                              const memory_desc_wrapper &weights_d,
                              const memory_desc_wrapper &dst_d, const primitive_attr_t &attr,
                              bool with_relu = false, double relu_negative_slope = 0.);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm, isa == avx2,
            Ymm, Zmm>::type;
    using reg64_t = const Reg64;

    const AddressFrame &vmmword = (isa == sse42) ? xword :
                                  (isa == avx2) ? yword : zword;

    const int vlen = cpu_isa_traits<isa>::vlen;

    reg64_t reg_input = rax;
    reg64_t aux_reg_input = r8;
    reg64_t reg_kernel = rdx;
    reg64_t aux_reg_kernel = r9;
    reg64_t reg_output = rsi;
    reg64_t reg_bias = rbx;
    reg64_t kj = r10;
    reg64_t oi_iter = r11;
    reg64_t ki_iter = r12;
    reg64_t reg_kh = abi_not_param1;
    reg64_t imm_addr64 = r13;
    reg64_t simd_iter = r15;

    Vmm vmm_res_ns = Vmm(1);
    Vmm vmm_mask = Vmm(0);

    Xmm xmm_relu_ns = Xmm(2);
    Vmm vmm_relu_ns = Vmm(2);
    Vmm vmm_zero = Vmm(3);

    Vmm get_ker_reg(int idx) { return Vmm(0 + idx); }
    Vmm get_src_reg(int idx) { return Vmm(1 + idx); }
    Vmm get_acc_reg(int idx) { return Vmm(4 + idx); }

    const unsigned char _cmp_gt_os = 6;
    const unsigned char _cmp_lt_os = 1;

    const unsigned char _op_floor = 1;

    Vmm vmm_src = Vmm(1);
    Vmm vmm_aux0 = Vmm(0);
    Vmm vmm_aux1 = Vmm(3);
    Vmm vmm_aux2 = Vmm(isa == avx512_common ? 31 : 0);

    Xbyak::Reg64 reg_table = aux_reg_input;

    inline void simd_expf(const Vmm &vmm_src);

    Xbyak::Label l_table;
    inline void prepare_table();

    inline void oh_step_unroll_kw(int ur_w, int pad_l, int pad_r);
    inline void oh_step_nopad(int ur_w, int pad_l, int pad_r, char pad_label);
    inline void width_blk_step(int ur_w, int pad_l, int pad_r, char pad_label);
    inline void solve_common();

    void generate();
};

template <cpu_isa_t isa>
struct jit_uni_dw_conv_bwd_data_kernel_f32: public jit_generator {
    jit_uni_dw_conv_bwd_data_kernel_f32(jit_conv_conf_t ajcp): jcp(ajcp)
    {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &diff_src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &diff_dst_d);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm, isa == avx2,
            Ymm, Zmm>::type;
    using reg64_t = const Xbyak::Reg64;

    int ker_offset = 0;
    int src_offset = isa == sse42 ? 2 : 1;
    int acc_offset = isa == sse42 ? 4 : 2;

    Vmm get_ker_reg(int idx) { return Vmm(ker_offset + idx); }
    Vmm get_src_reg(int idx) { return Vmm(src_offset + idx); }
    Vmm get_acc_reg(int idx) { return Vmm(acc_offset + idx); }

    reg64_t reg_ddst       = rax;
    reg64_t aux_reg_ddst   = r8;
    reg64_t aux1_reg_ddst = abi_not_param1;
    reg64_t reg_kernel     = rdx;
    reg64_t aux_reg_kernel = r10;
    reg64_t aux1_reg_kernel = rbp;
    reg64_t reg_dsrc       = rsi;

    reg64_t reg_ur_str_w = r9;
    reg64_t reg_oc_blocks = rbx;

    reg64_t iter_kh = r11;
    reg64_t iter_kw = r12;
    reg64_t reg_kh  = r13;
    reg64_t reg_kw  = r14;

    inline void kernel(int ur_w, int oc_blocks, char loop_type);

    void generate();
};

}
}
}

#endif
