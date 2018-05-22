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

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "cpu_memory.hpp"

#include "jit_uni_dw_conv_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

using namespace Xbyak;

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::oh_step_unroll_kw(int ur_w, int pad_l, int pad_r)
{
    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;

    for (int ki = 0; ki < kw; ki++) {
        int jj_start = nstl::max(0, div_up(pad_l - ki * dilate_w, stride_w));
        int jj_end = ur_w - nstl::max(0, div_up(ki*dilate_w+pad_r-(kw-1)*dilate_w, stride_w));
        int ker_off = ki * oc_blk;

        Vmm vmm_ker = get_ker_reg(0);
        uni_vmovups(vmm_ker, ptr[aux_reg_kernel + sizeof(float) * ker_off]);

        for (int jj = jj_start; jj < jj_end; jj++) {
            int inp_off = (ki*dilate_w + jj*stride_w - pad_l)*ic_blk;

            Vmm vmm_src = get_src_reg(0);
            uni_vmovups(vmm_src, ptr[aux_reg_input + sizeof(float) * inp_off]);

            Vmm vmm_acc = get_acc_reg(jj);
            uni_vfmadd231ps(vmm_acc, vmm_src, vmm_ker);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::oh_step_nopad(int ur_w,
        int pad_l, int pad_r, char pad_tag)
{
    jit_tagged_label kw_label("kw", pad_tag);

    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;

    xor_(ki_iter, ki_iter);
    L(kw_label);
    {
        Vmm vmm_ker = get_ker_reg(0);
        uni_vmovups(vmm_ker, ptr[aux_reg_kernel]);

        for (int jj = 0; jj < ur_w; jj++) {
            int inp_off = (jj * stride_w - pad_l) * ic_blk;

            Vmm vmm_src = get_src_reg(0);
            uni_vmovups(vmm_src, ptr[aux_reg_input + sizeof(float) * inp_off]);

            Vmm vmm_acc = get_acc_reg(jj);
            uni_vfmadd231ps(vmm_acc, vmm_src, vmm_ker);
        }

        add(aux_reg_kernel, sizeof(float) * oc_blk);
        add(aux_reg_input, sizeof(float) * ic_blk * dilate_w);

        inc(ki_iter);
        cmp(ki_iter, kw);
        jl(kw_label, T_NEAR);
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::width_blk_step(int ur_w,
        int pad_l, int pad_r, char pad_tag)
{
    int iw = jcp.iw;
    int kw = jcp.kw;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;
    const int inp_mult = ic_blk * dilate_h;
    const int inp_off = ic_blk * dilate_w;

    xor_(simd_iter, simd_iter);

    mov(aux_reg_input, reg_input);
    mov(aux_reg_kernel, reg_kernel);

    jit_tagged_label init_simd_iter_label("simd_iter", pad_tag);

    L(init_simd_iter_label);

    for (int jj = 0; jj < ur_w; jj++)
    {
        Vmm vmm_acc = get_acc_reg(jj);

        if (this->jcp.with_bias)
            uni_vmovups(vmm_acc, vmmword[reg_bias]);
        else
            uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
    }

    Label skip_kh_loop;
    mov(kj, reg_kh);
    if (jcp.kh <= jcp.t_pad) {
        cmp(kj, 0);
        je(skip_kh_loop, T_NEAR);
    }
    jit_tagged_label kh_label("kh", pad_tag);
    L(kh_label);
    {
        if (jcp.kw >= 5 && pad_l == 0 && pad_r == 0) {
            oh_step_nopad(ur_w, pad_l, pad_r, pad_tag);
            sub(aux_reg_input, sizeof(float) * kw * inp_off);
            add(aux_reg_input, sizeof(float) * iw * inp_mult);
        } else {
            oh_step_unroll_kw(ur_w, pad_l, pad_r);
            add(aux_reg_kernel, sizeof(float) * kw * oc_blk);
            add(aux_reg_input, sizeof(float) * iw * inp_mult);
        }

        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }

    L(skip_kh_loop);

    if (this->jcp.with_eltwise) {
        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
        if (jcp.eltwise_alpha == 0) {
            vmm_relu_ns = vmm_zero;
        } else {
            mov(imm_addr64, float2int(jcp.eltwise_alpha));
            movq(xmm_relu_ns, imm_addr64);
            uni_vbroadcastss(vmm_relu_ns, xmm_relu_ns);
        }

        if (this->jcp.eltwise_alg == mkldnn_eltwise_relu) {
            for (int jj = 0; jj < ur_w; jj++) {
                const size_t o_off = jj * oc_blk;
                Vmm vmm_dst = get_acc_reg(jj);

                if (isa == sse42) {
                    pxor(vmm_mask, vmm_mask);
                    cmpps(vmm_mask, vmm_dst, _cmp_gt_os);
                    movups(vmm_res_ns, vmm_dst);
                    mulps(vmm_res_ns, vmm_relu_ns);
                    blendvps(vmm_dst, vmm_res_ns);
                } else if (isa == avx2) {
                    vcmpgtps(vmm_mask, vmm_dst, vmm_zero);
                    vmulps(vmm_res_ns, vmm_relu_ns, vmm_dst);
                    vblendvps(vmm_dst, vmm_res_ns, vmm_dst, vmm_mask);
                } else if (isa == avx512_common) {
                    Opmask kmask = Opmask(7);
                    vcmpps(kmask, vmm_dst, vmm_zero, _cmp_lt_os);
                    vmulps(vmm_dst | kmask, vmm_dst, vmm_relu_ns);
                }

                uni_vmovups(vmmword[reg_output + sizeof(float) * o_off], vmm_dst);
            }
        } else if (this->jcp.eltwise_alg == mkldnn_eltwise_elu) {
            mov(reg_table, jit_uni_dw_conv_fwd_kernel_f32::l_table);

            for (int jj = 0; jj < ur_w; jj++) {
                const size_t o_off = jj * oc_blk;
                Vmm vmm_dst = get_acc_reg(jj);

                if (isa == sse42) {
                    movups(vmm_src, vmm_dst);
                    simd_expf(vmm_dst);
                    subps(vmm_dst, ptr[reg_table + 0 * vlen]);
                    mulps(vmm_dst, vmm_relu_ns);
                    pxor(vmm_mask, vmm_mask);
                    cmpps(vmm_mask, vmm_src, _cmp_gt_os);
                    blendvps(vmm_src, vmm_dst);
                    movups(vmmword[reg_output + sizeof(float) * o_off], vmm_src);
                } else if (isa == avx2) {
                    uni_vmovups(vmm_src, vmm_dst);
                    simd_expf(vmm_dst);
                    uni_vsubps(vmm_dst, vmm_dst, ptr[reg_table + 0 * vlen]);
                    uni_vmulps(vmm_dst, vmm_dst, vmm_relu_ns);
                    uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
                    uni_vcmpgtps(vmm_mask, vmm_src, vmm_zero);
                    uni_vblendvps(vmm_dst, vmm_dst, vmm_src, vmm_mask);
                    uni_vmovups(vmmword[reg_output + sizeof(float) * o_off], vmm_dst);
                } else if (isa == avx512_common) {
                    uni_vmovups(vmm_src, vmm_dst);
                    simd_expf(vmm_dst);
                    uni_vsubps(vmm_dst, vmm_dst, ptr[reg_table + 0 * vlen]);
                    uni_vmulps(vmm_dst, vmm_dst, vmm_relu_ns);

                    Opmask kmask = Opmask(1);
                    vpxord(vmm_zero, vmm_zero, vmm_zero);
                    vcmpps(kmask, vmm_src, vmm_zero, _cmp_gt_os);

                    vblendmps(vmm_dst | kmask, vmm_dst, vmm_src);
                    uni_vmovups(vmmword[reg_output + sizeof(float) * o_off], vmm_dst);
                }
            }
        }
    } else {
        for (int jj = 0; jj < ur_w; jj++) {
            const size_t o_off = jj * oc_blk;
            Vmm vmm_dst = get_acc_reg(jj);

            uni_vmovups(vmmword[reg_output + sizeof(float) * o_off], vmm_dst);
        }
    }

    if (isa == sse42) {
        mov(aux_reg_kernel, reg_kernel);
        mov(aux_reg_input, reg_input);
        add(aux_reg_input, sizeof(float) * 4);
        add(aux_reg_kernel, sizeof(float) * 4);
        add(reg_output, sizeof(float) * 4);
        add(reg_bias, sizeof(float) * 4);

        inc(simd_iter);
        cmp(simd_iter, 2);
        jl(init_simd_iter_label, T_NEAR);

        sub(reg_output, sizeof(float) * 8);
        sub(reg_bias, sizeof(float) * 8);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_fwd_kernel_f32<isa>::solve_common()
{
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int n_oi = jcp.ow / ur_w;
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;
    int dilate_w = jcp.dilate_w + 1;
    int str_w = jcp.stride_w;
    const int inp_mult = ic_blk;

    int l_pad = jcp.l_pad;
    int r_pad = nstl::max(0, (int(jcp.ow) - 1) * str_w + (kw - 1) * dilate_w
            - (iw + l_pad - 1));
    int r_pad1 = (ur_w * n_oi - 1) * str_w + (kw - 1) * dilate_w
            - (iw + l_pad - 1);
    if (r_pad1 > 0) n_oi--;

    if (l_pad > 0) {
        n_oi--;
        if (n_oi < 0 && r_pad1 > 0)
            width_blk_step(ur_w, l_pad, r_pad1, 'l'); // "lrpad"
        else
            width_blk_step(ur_w, l_pad, 0, 'l'); // "lpad"
        add(reg_input, sizeof(float) * (ur_w * str_w - l_pad) * inp_mult);
        add(reg_output, sizeof(float) * ur_w * oc_blk);
    }

    jit_tagged_label ow_loop_label("ow", '0');
    xor_(oi_iter, oi_iter);

    if (n_oi > 0) {
        L(ow_loop_label);

        width_blk_step(ur_w, 0, 0, 'm'); // "middle"
        add(reg_input, sizeof(float) * ur_w * str_w * inp_mult);
        add(reg_output, sizeof(float) * ur_w * oc_blk);

        inc(oi_iter);
        cmp(oi_iter, n_oi);
        jl(ow_loop_label, T_NEAR);
    }

    if (r_pad1 > 0 && n_oi >=0) {
        width_blk_step(ur_w, 0, r_pad1, 'r'); // "rpad"
        add(reg_input, sizeof(float) * ur_w * str_w * inp_mult);
        add(reg_output, sizeof(float) * ur_w * oc_blk);
    }

    if (ur_w_tail != 0)
        width_blk_step(ur_w_tail, 0, r_pad, 't'); // "tail"
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::generate()
{
    this->preamble();

    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    if (jcp.with_bias)
        mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);

    solve_common();

    this->postamble();

    prepare_table();
}

template <cpu_isa_t isa>
bool jit_uni_dw_conv_fwd_kernel_f32<isa>::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace primitive_kind;
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) {
        return p.entry_[idx].kind == eltwise
            && p.entry_[idx].eltwise.scale == 1.
            && (p.entry_[idx].eltwise.alg == alg_kind::eltwise_relu ||
                (p.entry_[idx].eltwise.alg == alg_kind::eltwise_elu && isa != avx512_common));
    };

    switch (p.len_) {
    case 0: return true; // no post_ops
    case 1: return true // sum OR eltwise
                && !jcp.with_eltwise
                && (is_eltwise(0) || p.contain(sum, 0));
    case 2: return true // sum->eltwise
                && !jcp.with_eltwise
                && (p.contain(sum, 0) && is_eltwise(1));
    default: return false;
    }

    return false;
}

template <cpu_isa_t isa>
status_t jit_uni_dw_conv_fwd_kernel_f32<isa>::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr, bool with_relu, double relu_negative_slope)
{
    if (!mayiuse(isa)) return status::unimplemented;

    jcp.prop_kind = cd.prop_kind;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    if (!with_groups) return status::unimplemented;

    jcp.ngroups = weights_d.dims()[0];
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1];
    jcp.ic = src_d.dims()[1];

    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2];
    jcp.ow = dst_d.dims()[3];

    jcp.kh = weights_d.dims()[3];
    jcp.kw = weights_d.dims()[4];

    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.b_pad = cd.padding[1][0];
    jcp.r_pad = cd.padding[1][1];

    if ((jcp.t_pad == 0 && jcp.b_pad != 0) ||
        (jcp.l_pad == 0 && jcp.r_pad != 0))
        return status::unimplemented;

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    jcp.src_fmt = src_d.format();
    jcp.with_bias = cd.bias_desc.format != memory_format::undef;
    jcp.with_eltwise = with_relu;
    jcp.eltwise_alg = mkldnn_eltwise_relu;
    jcp.eltwise_alpha = relu_negative_slope;

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    if (!jcp.with_eltwise) {
        int eltwise_ind = p.find(primitive_kind::eltwise);
        if (eltwise_ind != -1) {
            jcp.with_eltwise  = true;
            jcp.eltwise_alg   = p.entry_[eltwise_ind].eltwise.alg;
            jcp.eltwise_alpha = p.entry_[eltwise_ind].eltwise.alpha;
            jcp.eltwise_beta  = p.entry_[eltwise_ind].eltwise.beta;
            jcp.eltwise_scale = p.entry_[eltwise_ind].eltwise.scale;
        }
    }

    auto desired_act_fmt = isa == avx512_common ? nChw16c : nChw8c;
    auto desired_wei_fmt = isa == avx512_common ? Goihw16g : Goihw8g;

    bool args_ok = true
        && jcp.oc == jcp.ngroups
        && jcp.ic == jcp.ngroups
        && src_d.format() == desired_act_fmt
        && weights_d.format() == desired_wei_fmt
        && one_of(cd.bias_desc.format, memory_format::undef, any, x)
        && dst_d.format() == desired_act_fmt;
    if (!args_ok) return status::unimplemented;

    const int simd_w = isa == avx512_common ? 16 : 8;

    jcp.ur_h = 1; /* no code-unrolling by h so far */
    jcp.ur_w = isa == avx512_common ? 27 : 12;
    if (jcp.ow < jcp.ur_w) jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    args_ok = true
        && jcp.ic == jcp.oc
        && jcp.oc % simd_w == 0
        && jcp.l_pad <= jcp.ur_w
        && implication(jcp.kw > 7, (jcp.t_pad == 0 && jcp.l_pad == 0)
                || (jcp.stride_w == 1 && jcp.stride_h == 1));
    if (!args_ok) return status::unimplemented;

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
        + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));

    if (r_pad_no_tail > jcp.ur_w) {
        /* recalculate ur_w, nb_oc_blocking and ur_w_tail */
        jcp.ur_w = r_pad_no_tail + 1;
        jcp.ur_w_tail = jcp.ow % jcp.ur_w;
        /* check again ... */
        r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));

        if ((r_pad_no_tail > jcp.ur_w) || (jcp.ow < jcp.ur_w) || (jcp.ur_w > 12))
            return status::unimplemented;
    }
    if (jcp.l_pad > jcp.ur_w) return status::unimplemented;

    jcp.ic_block = simd_w;

    jcp.oc_block = simd_w;
    jcp.nb_oc = jcp.oc / jcp.oc_block;


    return status::success;
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::simd_expf(const Vmm &vmm_src) {
    uni_vminps(vmm_src, vmm_src, ptr[reg_table + 10 * vlen]);
    uni_vmaxps(vmm_src, vmm_src, ptr[reg_table + 11 * vlen]);
    uni_vmovups(vmm_aux0, vmm_src);
    //calculate exp(x)
    // fx = x * log2ef + 0.5
    uni_vmulps(vmm_src, vmm_src, ptr[reg_table + 2 * vlen]);
    uni_vaddps(vmm_src, vmm_src, ptr[reg_table + 1 * vlen]);

    // tmp = floorf(fx)
    if (isa == avx512_common) {
        vcvtps2dq(vmm_aux1 | T_rd_sae, vmm_src);
        vcvtdq2ps(vmm_aux1, vmm_aux1);

        unsigned char _cmp_gt_os = 14;
        Xbyak::Opmask k_mask_tmp = Xbyak::Opmask(2);
        vcmpps(k_mask_tmp, vmm_aux1, vmm_src, _cmp_gt_os);
        vmovups(vmm_aux2 | k_mask_tmp | T_z, zword[reg_table + 0 * vlen]);

        uni_vsubps(vmm_aux1, vmm_aux1, vmm_aux2);
    } else {
        uni_vroundps(vmm_aux1, vmm_src, _op_floor);
    }

    //keep fx for further computations
    uni_vmovups(vmm_src, vmm_aux1); //vmm_src = fx

    //x = x - fx * ln2
    uni_vfnmadd231ps(vmm_aux0, vmm_aux1, ptr[reg_table + 3 * vlen]);

    // compute 2^n
    uni_vcvtps2dq(vmm_aux1, vmm_src);
    uni_vpaddd(vmm_aux1, vmm_aux1, ptr[reg_table + 4 * vlen]);
    uni_vpslld(vmm_aux1, vmm_aux1, 23); //Vmm(6) = 2^-fx

    // y = p5
    uni_vmovups(vmm_src, ptr[reg_table + 9 * vlen]);
    // y = y * x + p4
    uni_vfmadd213ps(vmm_src, vmm_aux0, ptr[reg_table + 8 * vlen]);
    // y = y * x + p3
    uni_vfmadd213ps(vmm_src, vmm_aux0, ptr[reg_table + 7 * vlen]);
    // y = y * x + p2
    uni_vfmadd213ps(vmm_src, vmm_aux0, ptr[reg_table + 6 * vlen]);
    // y = y * x + p1
    uni_vfmadd213ps(vmm_src, vmm_aux0, ptr[reg_table + 0 * vlen]);
    // y = y * x + p0
    uni_vfmadd213ps(vmm_src, vmm_aux0, ptr[reg_table + 5 * vlen]);  //exp(q)
    // y = y * 2^n
    uni_vmulps(vmm_src, vmm_src, vmm_aux1);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::prepare_table() {
    const unsigned int cvals[] = {
            0x3f800000, // [0] 1.0f
            0x3f000000, // [1] 0.5f
            0x3fb8aa3b, // [2] log2ef = 1.44269502f
            0x3f317218, // [3] ln2f =   0.69314718f
            0x0000007f, // [4] 0x7f
            // exp(x) polynom
            0x3f800001, // [5] p0 = 1.0000001f
            0x3efffe85, // [6] p2 = 0.4999887f
            0x3e2aaa3e, // [7] p3 = 0.16666505f
            0x3d2bb1b1, // [8] p4 = 0.041917507f
            0x3c091ec1, // [9] p5 = 0.008369149f
            0x42b0c0a5, //[10] max logf = 88.3762589f
            0xc1766666  //[11] min logf = -14.5f
    };

    align(64);
    L(l_table);
    for (size_t i = 0; i < sizeof(cvals) / sizeof(cvals[0]); ++i) {
        for (size_t d = 0; d < vlen / sizeof(float); ++d) {
            dd(cvals[i]);
        }
    }
}

template struct jit_uni_dw_conv_fwd_kernel_f32<avx512_common>;
template struct jit_uni_dw_conv_fwd_kernel_f32<avx2>;
template struct jit_uni_dw_conv_fwd_kernel_f32<sse42>;

template <cpu_isa_t isa>
void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::kernel(int ur_str_w, int oc_blocks, char loop_type_tag)
{
    jit_tagged_label iter_exit_label("iter_exit", ur_str_w, loop_type_tag);
    jit_tagged_label kh_label("kh", ur_str_w, loop_type_tag);
    jit_tagged_label kw_label("kw", ur_str_w, loop_type_tag);

    int kw = jcp.kw;
    int kh = jcp.kh;
    int ow = jcp.ow;
    int oh = jcp.oh;

    int iw = jcp.iw;
    int ih = jcp.ih;

    int oc_blk = jcp.oc_block;
    int ic_blk = jcp.ic_block;
    int stride_h = jcp.stride_h;
    int stride_w = jcp.stride_w;

    for (int c = 0; c < oc_blocks; c++) {
        for (int w = 0; w < ur_str_w; w++) {
            Vmm vmm_acc = get_acc_reg(c*ur_str_w + w);
            uni_vpxor(vmm_acc, vmm_acc, vmm_acc);

            if (isa == sse42) {
                Vmm vmm_acc2 = get_acc_reg(oc_blocks*ur_str_w + c*ur_str_w + w);
                uni_vpxor(vmm_acc2, vmm_acc2, vmm_acc2);
            }
        }
    }

    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);

    cmp(reg_kw, 0);
    je(iter_exit_label, T_NEAR);

    mov(iter_kh, reg_kh);
    L(kh_label); {

        mov(aux1_reg_ddst, aux_reg_ddst);
        mov(aux1_reg_kernel, aux_reg_kernel);

        mov(iter_kw, reg_kw);
        L(kw_label); {

            if (isa == sse42) {
                for (int c = 0; c < oc_blocks; c++) {
                    uni_vmovups(get_ker_reg(0), ptr[aux1_reg_kernel + (c * kh * kw * oc_blk + 0) * sizeof(float)]);
                    uni_vmovups(get_ker_reg(1), ptr[aux1_reg_kernel + (c * kh * kw * oc_blk + 4) * sizeof(float)]);

                    for (int w = 0; w < ur_str_w; w++) {
                        Vmm vmm_acc = get_acc_reg(c * ur_str_w + w);

                        uni_vmovups(get_src_reg(0), ptr[aux1_reg_ddst + ((c * oh * ow + w) * oc_blk + 0) * sizeof(float)]);
                        uni_vmovups(get_src_reg(1), ptr[aux1_reg_ddst + ((c * oh * ow + w) * oc_blk + 4) * sizeof(float)]);
                        uni_vfmadd231ps(get_acc_reg(c * ur_str_w + w), get_src_reg(0), get_ker_reg(0));
                        uni_vfmadd231ps(get_acc_reg(oc_blocks*ur_str_w + c * ur_str_w + w), get_src_reg(1), get_ker_reg(1));
                    }
                }
            } else {
                for (int c = 0; c < oc_blocks; c++) {
                    Vmm vmm_ker = get_ker_reg(0);
                    uni_vmovups(vmm_ker, ptr[aux1_reg_kernel + c * kh * kw * oc_blk * sizeof(float)]);

                    for (int w = 0; w < ur_str_w; w++) {
                        Vmm vmm_src = get_src_reg(0);
                        Vmm vmm_acc = get_acc_reg(c * ur_str_w + w);

                        uni_vmovups(vmm_src, ptr[aux1_reg_ddst + (c * oh * ow + w) * oc_blk * sizeof(float)]);
                        uni_vfmadd231ps(vmm_acc, vmm_src, vmm_ker);
                    }
                }
            }

            add(aux1_reg_kernel, sizeof(float) * oc_blk * stride_w);
            sub(aux1_reg_ddst, sizeof(float) * oc_blk);

            sub(iter_kw, stride_w);
            cmp(iter_kw, 0);
            jg(kw_label, T_NEAR);
        }

        add(aux_reg_kernel, sizeof(float) * kw * oc_blk * stride_h);
        sub(aux_reg_ddst, sizeof(float) * ow * oc_blk);

        sub(iter_kh, stride_h);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);

    for (int c = 0; c < oc_blocks; c++) {
        for (int w = 0; w < ur_str_w; w++) {
            Vmm vmm_acc = get_acc_reg(c*ur_str_w + w);
            uni_vmovups(ptr[reg_dsrc + ((c*ih*iw + w*stride_w)*ic_blk + 0) * sizeof(float)], vmm_acc);

            if (isa == sse42) {
                Vmm vmm_acc2 = get_acc_reg(oc_blocks*ur_str_w + c*ur_str_w + w);
                uni_vmovups(ptr[reg_dsrc + ((c*ih*iw + w*stride_w)*ic_blk + 4) * sizeof(float)], vmm_acc2);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::generate() {
    preamble();

    const char *tail_label = "tail";
    const char *exit_label = "exit";

    auto loop_body = [=] (int oc_blocks, char loop_type_tag) {
        jit_tagged_label loop_ur_str_w_4("loop_ur_str_w_4", loop_type_tag);
        jit_tagged_label loop_ur_str_w_3("loop_ur_str_w_3", loop_type_tag);
        jit_tagged_label loop_ur_str_w_2("loop_ur_str_w_2", loop_type_tag);
        jit_tagged_label loop_ur_str_w_1("loop_ur_str_w_1", loop_type_tag);

        if (jcp.ur_w >= 4) {
            L(loop_ur_str_w_4);
            {
                cmp(reg_ur_str_w, 4);
                jl(loop_ur_str_w_3, T_NEAR);

                mov(aux_reg_ddst, reg_ddst);
                mov(aux_reg_kernel, reg_kernel);

                kernel(4, oc_blocks, loop_type_tag);

                add(reg_dsrc, sizeof(float) * 4 * jcp.oc_block * jcp.stride_w);
                add(reg_ddst, sizeof(float) * 4 * jcp.oc_block);

                sub(reg_ur_str_w, 4);
                jmp(loop_ur_str_w_4);
            }
        }

        if (jcp.ur_w >= 3) {
            L(loop_ur_str_w_3); {
                cmp(reg_ur_str_w, 3);
                jl(loop_ur_str_w_2, T_NEAR);

                mov(aux_reg_ddst, reg_ddst);
                mov(aux_reg_kernel, reg_kernel);

                kernel(3, oc_blocks, loop_type_tag);

                add(reg_dsrc, sizeof(float) * 3 * jcp.oc_block * jcp.stride_w);
                add(reg_ddst, sizeof(float) * 3 * jcp.oc_block);

                sub(reg_ur_str_w, 3);
                jmp(loop_ur_str_w_3);
            }
        }

        if (jcp.ur_w >= 2) {
            L(loop_ur_str_w_2); {
                cmp(reg_ur_str_w, 2);
                jl(loop_ur_str_w_1, T_NEAR);

                mov(aux_reg_ddst, reg_ddst);
                mov(aux_reg_kernel, reg_kernel);

                kernel(2, oc_blocks, loop_type_tag);

                add(reg_dsrc, sizeof(float) * 2 * jcp.oc_block * jcp.stride_w);
                add(reg_ddst, sizeof(float) * 2 * jcp.oc_block);

                sub(reg_ur_str_w, 2);
                jmp(loop_ur_str_w_2);
            }
        }

        L(loop_ur_str_w_1); {
            cmp(reg_ur_str_w, 1);
            jl(exit_label, T_NEAR);

            mov(aux_reg_ddst, reg_ddst);
            mov(aux_reg_kernel, reg_kernel);

            kernel(1, oc_blocks, loop_type_tag);

            add(reg_dsrc, sizeof(float) * 1 * jcp.oc_block * jcp.stride_w);
            add(reg_ddst, sizeof(float) * 1 * jcp.oc_block);
        }
    };

    mov(reg_dsrc, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_ddst, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_kw, ptr[this->param1 + GET_OFF(kw_padding)]);
    mov(reg_oc_blocks, ptr[this->param1 + GET_OFF(oc_blocks)]);
    mov(reg_ur_str_w, ptr[this->param1 + GET_OFF(ur_str_w)]);

    int nb_oc_tail = jcp.nb_oc % jcp.nb_oc_blocking;

    cmp(reg_oc_blocks, jcp.nb_oc_blocking);
    jne(nb_oc_tail ? tail_label : exit_label, T_NEAR);

    loop_body(jcp.nb_oc_blocking, 'm'); // channel main loop

    if (nb_oc_tail) {
        L(tail_label);

        cmp(reg_oc_blocks, nb_oc_tail);
        jne(exit_label, T_NEAR);

        loop_body(nb_oc_tail, 't'); // channel tail loop
    }

    L(exit_label);

    this->postamble();
}

template <cpu_isa_t isa>
status_t jit_uni_dw_conv_bwd_data_kernel_f32<isa>::init_conf(jit_conv_conf_t &jcp,
                                                             const convolution_desc_t &cd, const memory_desc_wrapper &diff_src_d,
                                                             const memory_desc_wrapper &weights_d,
                                                             const memory_desc_wrapper &diff_dst_d)
{
    if (!mayiuse(isa)) return status::unimplemented;

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    if (!with_groups) return status::unimplemented;

    jcp.ngroups = weights_d.dims()[0];
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1];
    jcp.ic = diff_src_d.dims()[1];

    jcp.ih = diff_src_d.dims()[2];
    jcp.iw = diff_src_d.dims()[3];
    jcp.oh = diff_dst_d.dims()[2];
    jcp.ow = diff_dst_d.dims()[3];

    jcp.kh = weights_d.dims()[3];
    jcp.kw = weights_d.dims()[4];

    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.b_pad = cd.padding[1][0];
    jcp.r_pad = cd.padding[1][1];

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    const int simd_w = isa == avx512_common ? 16 : 8;

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;

    jcp.ic_block = simd_w;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    jcp.oc_block = simd_w;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.src_fmt = diff_src_d.format();

    auto desired_act_fmt = isa == avx512_common ? nChw16c : nChw8c;
    auto desired_wei_fmt = isa == avx512_common ? Goihw16g : Goihw8g;

    bool args_ok = true
                   && diff_src_d.format() == desired_act_fmt
                   && weights_d.format() == desired_wei_fmt
                   && diff_dst_d.format() == desired_act_fmt
                   && jcp.ngroups % simd_w == 0
                   && jcp.oc == jcp.ngroups
                   && jcp.ic == jcp.ngroups
                   && jcp.dilate_h == 0
                   && jcp.dilate_w == 0
                   && jcp.oh == (jcp.ihp - jcp.kh) / jcp.stride_h + 1
                   && jcp.ow == (jcp.iwp - jcp.kw) / jcp.stride_w + 1;
    if (!args_ok) return status::unimplemented;

    jcp.ur_h = 1;

    jcp.ur_w = isa == sse42 ? 2 : 4;
    jcp.nb_oc_blocking = isa != avx512_common ? 3 : 7;

    return status::success;
}

template struct jit_uni_dw_conv_bwd_data_kernel_f32<avx512_common>;
template struct jit_uni_dw_conv_bwd_data_kernel_f32<avx2>;
template struct jit_uni_dw_conv_bwd_data_kernel_f32<sse42>;

}
}
}
