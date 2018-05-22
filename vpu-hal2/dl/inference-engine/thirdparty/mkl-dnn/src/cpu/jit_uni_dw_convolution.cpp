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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "jit_uni_dw_convolution.hpp"
#include "mkldnn_thread.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

template <cpu_isa_t isa, bool with_relu>
void _jit_uni_dw_convolution_fwd_t<isa, with_relu>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const auto &jcp = kernel_->jcp;

    int ocb_work = jcp.nb_oc;
    int MB = jcp.mb;
    const size_t work_amount = MB * ocb_work * jcp.oh;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        size_t n{0}, ch{0}, oh{0};
        nd_iterator_init(start, n, MB, ch, ocb_work, oh, jcp.oh);
        for (size_t iwork = start; iwork < end; ++iwork) {
            jit_conv_call_s par_conv = {};

            const int ij = oh * jcp.stride_h;
            const int i_t_overflow = nstl::max(0, jcp.t_pad - ij);
            const int i_b_overflow = nstl::max(jcp.ih, ij
                + (jcp.kh-1) * (jcp.dilate_h+1) - jcp.t_pad+1) - jcp.ih;

            const int ih = nstl::max(ij - jcp.t_pad + div_up(i_t_overflow, (jcp.dilate_h+1)) * (jcp.dilate_h + 1), 0);
            par_conv.src = &src[src_d.blk_off(n, ch, ih, 0)];
            par_conv.dst = &dst[dst_d.blk_off(n, ch, oh, 0)];

            const int wh = div_up(i_t_overflow, (jcp.dilate_h + 1));
            par_conv.filt = &weights[weights_d.blk_off(ch, 0, 0, wh, 0)];

            if (bias) par_conv.bias = &bias[bias_d.blk_off(ch * jcp.oc_block)];

            par_conv.oc_blocks = nstl::min((int)ch + 1, jcp.nb_oc) - ch;

            par_conv.kw_padding = 0;
            const int kh_padding = jcp.kh
                - div_up(i_t_overflow, (jcp.dilate_h + 1))
                - div_up(i_b_overflow, (jcp.dilate_h + 1));
            par_conv.kh_padding = nstl::max(0, kh_padding);
            kernel_->jit_ker(&par_conv);

            nd_iterator_step(n, MB, ch, ocb_work, oh, jcp.oh);
        }
    };

    #pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}

template void _jit_uni_dw_convolution_fwd_t<avx512_common, false>::execute_forward();
template void _jit_uni_dw_convolution_fwd_t<avx2, false>::execute_forward();
template void _jit_uni_dw_convolution_fwd_t<sse42, false>::execute_forward();

template void _jit_uni_dw_convolution_fwd_t<avx512_common, true>::execute_forward();
template void _jit_uni_dw_convolution_fwd_t<avx2, true>::execute_forward();
template void _jit_uni_dw_convolution_fwd_t<sse42, true>::execute_forward();

template <cpu_isa_t isa>
void _jit_uni_dw_convolution_bwd_data_t<isa>::execute_backward_data() {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_src_d(conf_.diff_src_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));

    const auto &jcp = kernel_->jcp;

    int icb_work = utils::div_up(jcp.nb_oc, jcp.nb_oc_blocking);
    int MB = jcp.mb;
    const size_t work_amount = MB * icb_work * jcp.ih;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        size_t n{0}, cb{0}, ih{0};
        nd_iterator_init(start, n, MB, cb, icb_work, ih, jcp.ih);
        for (size_t iwork = start; iwork < end; ++iwork) {
            int ch = cb * jcp.nb_oc_blocking;
            int ch_num = jcp.nb_oc_blocking;

            jit_conv_call_s par_conv = {};
            par_conv.oc_blocks = nstl::min(ch + ch_num, jcp.nb_oc) - ch;

            const int i_t_overflow = nstl::max(0, (int)(jcp.kh - 1 - ih - jcp.t_pad));
            const int i_b_overflow = nstl::max(0, (int)(jcp.kh - 1 - (jcp.ih - 1 - ih) - jcp.b_pad));

            int oh = ih + jcp.t_pad - i_b_overflow;
            int stride_off_h = oh % jcp.stride_h;
            oh /= jcp.stride_h;

            for (int i_str_w = 0; i_str_w < jcp.stride_w; i_str_w++) {

                int iw = i_str_w;
                int l_border = nstl::min(jcp.kw - 1 - jcp.l_pad, jcp.iw);
                for (; iw < l_border; iw += jcp.stride_w) {
                    const int i_l_overflow = nstl::max(0, (jcp.kw - 1 - iw - jcp.l_pad));
                    const int i_r_overflow = nstl::max(0, (jcp.kw - 1 - (jcp.iw - 1 - iw) - jcp.r_pad));

                    int ow = iw + jcp.l_pad - i_r_overflow;
                    int stride_off_w = ow % jcp.stride_w;
                    ow /= jcp.stride_w;

                    par_conv.src = &diff_src[diff_src_d.blk_off(n, ch, ih, iw)];
                    par_conv.dst = &diff_dst[diff_dst_d.blk_off(n, ch, oh, ow)];
                    par_conv.filt = &weights[weights_d.blk_off(ch, 0, 0, i_b_overflow + stride_off_h,
                                                               i_r_overflow + stride_off_w)];

                    par_conv.kh_padding = nstl::max(0, jcp.kh - i_t_overflow - i_b_overflow - stride_off_h);
                    par_conv.kw_padding = nstl::max(0, jcp.kw - i_l_overflow - i_r_overflow - stride_off_w);

                    par_conv.ur_str_w = 1;

                    kernel_->jit_ker(&par_conv);
                }

                {
                    int r_border = jcp.iw - jcp.kw + jcp.r_pad;
                    int general_step = (r_border - iw) / jcp.stride_w;

                    int ow = iw + jcp.l_pad;
                    int stride_off_w = ow % jcp.stride_w;
                    ow /= jcp.stride_w;

                    par_conv.src = &diff_src[diff_src_d.blk_off(n, ch, ih, iw)];
                    par_conv.dst = &diff_dst[diff_dst_d.blk_off(n, ch, oh, ow)];
                    par_conv.filt = &weights[weights_d.blk_off(ch, 0, 0, i_b_overflow + stride_off_h, stride_off_w)];

                    par_conv.kh_padding = nstl::max(0, jcp.kh - i_t_overflow - i_b_overflow - stride_off_h);
                    par_conv.kw_padding = nstl::max(0, jcp.kw - stride_off_w);

                    par_conv.ur_str_w = general_step;

                    kernel_->jit_ker(&par_conv);

                    iw += general_step * jcp.stride_w;

                }

                for (; iw < jcp.iw; iw += jcp.stride_w) {
                    const int i_l_overflow = nstl::max(0, (jcp.kw - 1 - iw - jcp.l_pad));
                    const int i_r_overflow = nstl::max(0, (jcp.kw - 1 - (jcp.iw - 1 - iw) - jcp.r_pad));

                    int ow = iw + jcp.l_pad - i_r_overflow;
                    int stride_off_w = ow % jcp.stride_w;
                    ow /= jcp.stride_w;

                    par_conv.src = &diff_src[diff_src_d.blk_off(n, ch, ih, iw)];
                    par_conv.dst = &diff_dst[diff_dst_d.blk_off(n, ch, oh, ow)];
                    par_conv.filt = &weights[weights_d.blk_off(ch, 0, 0, i_b_overflow + stride_off_h,
                                                               i_r_overflow + stride_off_w)];

                    par_conv.kh_padding = nstl::max(0, jcp.kh - i_t_overflow - i_b_overflow - stride_off_h);
                    par_conv.kw_padding = nstl::max(0, jcp.kw - i_l_overflow - i_r_overflow - stride_off_w);

                    par_conv.ur_str_w = 1;

                    kernel_->jit_ker(&par_conv);
                }
            }
            nd_iterator_step(n, MB, cb, icb_work, ih, jcp.ih);
        }
    };

    #pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}

template void _jit_uni_dw_convolution_bwd_data_t<avx512_common>::execute_backward_data();
template void _jit_uni_dw_convolution_bwd_data_t<avx2>::execute_backward_data();
template void _jit_uni_dw_convolution_bwd_data_t<sse42>::execute_backward_data();

}
}
}