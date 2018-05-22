/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"
#include "mkldnn.hpp"

namespace mkldnn {

template <typename data_t>
void check_roi_pool_fwd(roi_pool_test_params p, const memory &input_data, const memory &input_roi, memory &output)
{
    const data_t* src_data = (data_t *)input_data.get_data_handle();
    const data_t* src_roi = (data_t *)input_roi.get_data_handle();
    data_t* dst = (data_t *)output.get_data_handle();

    const memory::desc src_d = input_data.get_primitive_desc().desc();
    const memory::desc src_roi_d = input_roi.get_primitive_desc().desc();
    const memory::desc dst_d = output.get_primitive_desc().desc();

    int C = p.test_pd.data.c;
    int H = p.test_pd.data.h;
    int W = p.test_pd.data.w;

    int ROIS = p.test_pd.roi.mb;

    double spatial_scale = p.test_pd.spatial_scale;
    int pooled_h = p.test_pd.pooled_h;
    int pooled_w = p.test_pd.pooled_w;

    for (int i = 0; i < ROIS * C * pooled_h * pooled_w; i++) {
        dst[i] = -FLT_MAX;
    }

    for (int n = 0; n < ROIS; ++n) {
        int roi_idx = p.roi_format == mkldnn::memory::format::nc ? n * p.test_pd.roi.c
                    : n * p.test_pd.roi.c * p.test_pd.roi.h * p.test_pd.roi.w;
        const data_t* src_roi_ptr = src_roi + map_index(src_roi_d, roi_idx);

        int roi_batch_ind = src_roi_ptr[0];
        int roi_start_w = round(src_roi_ptr[1] * spatial_scale);
        int roi_start_h = round(src_roi_ptr[2] * spatial_scale);
        int roi_end_w = round(src_roi_ptr[3] * spatial_scale);
        int roi_end_h = round(src_roi_ptr[4] * spatial_scale);

        int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
        int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);

        for (int c = 0; c < C; ++c) {

            for (int ph = 0; ph < pooled_h; ++ph) {
                for (int pw = 0; pw < pooled_w; ++pw) {
                    int hstart = (ph * roi_height) / pooled_h;
                    if ( (hstart * pooled_h) > (ph * roi_height) ) {
                        --hstart;
                    }

                    int wstart = (pw * roi_width) / pooled_w;
                    if ( (wstart * pooled_w) > (pw * roi_width) ) {
                        --wstart;
                    }

                    int hend = ((ph + 1) * roi_height) / pooled_h;
                    if ( (hend * pooled_h) < ((ph + 1) * roi_height) ) {
                        ++hend;
                    }

                    int wend = ((pw + 1) * roi_width) / pooled_w;
                    if ( (wend * pooled_w) < ((pw + 1) * roi_width) ) {
                        ++wend;
                    }

                    hstart = std::min(std::max(hstart + roi_start_h, 0), H);
                    hend = std::min(std::max(hend + roi_start_h, 0), H);
                    wstart = std::min(std::max(wstart + roi_start_w, 0), W);
                    wend = std::min(std::max(wend + roi_start_w, 0), W);

                    bool is_empty = (hend <= hstart) || (wend <= wstart);

                    int dst_idx = n * p.test_pd.data.c * p.test_pd.pooled_h * p.test_pd.pooled_w
                                + c * p.test_pd.pooled_h * p.test_pd.pooled_w + ph * p.test_pd.pooled_w + pw;
                    int pool_index = map_index(dst_d, dst_idx);

                    if (is_empty) {
                        dst[pool_index] = 0;
                    }

                    for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                            int src_idx = roi_batch_ind * p.test_pd.data.c * p.test_pd.data.h * p.test_pd.data.w
                                        + c * p.test_pd.data.h * p.test_pd.data.w + h * p.test_pd.data.w + w;
                            data_t batch_data = src_data[map_index(src_d, src_idx)];

                            if (batch_data > dst[pool_index]) {
                                dst[pool_index] = batch_data;
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename data_t>
class roi_pooling_test : public ::testing::TestWithParam<roi_pool_test_params> {
protected:
    virtual void SetUp()
    {
        roi_pool_test_params p = ::testing::TestWithParam<roi_pool_test_params>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_TRUE(p.aprop_kind == prop_kind::forward_inference);

        auto eng = engine(p.engine_kind, 0);
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, mkldnn::memory::data_type::f32);

        test_roi_pool_desc_t pd = p.test_pd;

        auto p_src_data = create_md({ pd.data.mb, pd.data.c, pd.data.h, pd.data.w }, data_type, p.data_format);

        auto p_src_roi = p.roi_format == mkldnn::memory::format::nc ?
            create_md({ pd.roi.mb, pd.roi.c}, data_type, p.roi_format):
            create_md({ pd.roi.mb, pd.roi.c, pd.roi.h, pd.roi.w }, data_type, p.roi_format);

        auto p_dst_desc = create_md({ pd.roi.mb, pd.data.c, pd.pooled_h, pd.pooled_w }, data_type, p.dst_format);

        std::vector<memory::desc> src_d;
        src_d.push_back(p_src_data);
        src_d.push_back(p_src_roi);

        auto roi_pool_desc = roi_pooling_forward::desc(p.aprop_kind, src_d, p_dst_desc, pd.pooled_h, pd.pooled_w, pd.spatial_scale);

        auto roi_pool_prim_desc = roi_pooling_forward::primitive_desc(roi_pool_desc, eng);

        auto src_data = memory({p_src_data, eng});
        auto src_roi = memory({p_src_roi, eng});
        auto dst_data = memory({p_dst_desc, eng});

        fill_data<data_t>(src_data.get_primitive_desc().get_size() / sizeof(data_t), (data_t *)src_data.get_data_handle());

        ASSERT_EQ(pd.roi.c, 5);
        for (int i = 0; i < pd.roi.mb; i++)
        {
            int roi_off = p.roi_format == mkldnn::memory::format::nc ? i * pd.roi.c : i * pd.roi.c * pd.roi.h * pd.roi.w;
            data_t* proi = ((data_t *)src_roi.get_data_handle() + roi_off);
            proi[0] = int(rand() % pd.data.mb);
            proi[1] = data_t(rand() % pd.data.w) / pd.spatial_scale;
            proi[2] = data_t(rand() % pd.data.h) / pd.spatial_scale;
            proi[3] = data_t(rand() % pd.data.w) / pd.spatial_scale;
            proi[4] = data_t(rand() % pd.data.h) / pd.spatial_scale;
        }

        std::vector<primitive::at> src_d_p;
        src_d_p.push_back(src_data);
        src_d_p.push_back(src_roi);

        auto roi_pool = roi_pooling_forward(roi_pool_prim_desc, src_d_p, dst_data);

        stream(stream::kind::lazy).submit({roi_pool}).wait();

        size_t roi_pooling_ref_size = pd.roi.mb * pd.data.c * pd.pooled_h * pd.pooled_w;
        float* roi_pooling_ref = new float[roi_pooling_ref_size];
        auto roi_pooling_ref_memory = memory(memory::primitive_desc(p_dst_desc, eng), roi_pooling_ref);
        check_roi_pool_fwd<data_t>(p, src_data, src_roi, roi_pooling_ref_memory);

        compare_data<data_t>(roi_pooling_ref_memory, dst_data);
    }
};

using roi_pooling_test_float = roi_pooling_test<float>;
using roi_pool_test_params_float = roi_pool_test_params;

TEST_P(roi_pooling_test_float, TestsROIPooling)
{
}

INSTANTIATE_TEST_CASE_P(
    TestROIPoolingForward, roi_pooling_test_float, ::testing::Values(
        roi_pool_test_params_float{ prop_kind::forward_inference,
        engine::kind::cpu, memory::format::nchw, memory::format::nc,
            memory::format::nchw, { { 1, 2, 100, 100 }, { 2, 5 }, 10, 10, 0.0625 } },
        roi_pool_test_params_float{ prop_kind::forward_inference,
        engine::kind::cpu, memory::format::nchw, memory::format::nchw,
            memory::format::nchw, { { 1, 2, 100, 100 }, { 2, 5, 1, 1 }, 10, 10, 0.0625 } },
        roi_pool_test_params_float{ prop_kind::forward_inference,
        engine::kind::cpu, memory::format::nChw8c, memory::format::nc,
            memory::format::nChw8c, { { 1, 64, 100, 100 }, { 1, 5 }, 10, 10, 0.0625 } },
        roi_pool_test_params_float{ prop_kind::forward_inference,
        engine::kind::cpu, memory::format::nChw8c, memory::format::nchw,
            memory::format::nChw8c, { { 1, 64, 100, 100 }, { 1, 5, 1, 1 }, 10, 10, 0.0625 } },
        roi_pool_test_params_float{ prop_kind::forward_inference,
        engine::kind::cpu, memory::format::nChw8c, memory::format::nc,
            memory::format::nChw8c, { { 1, 48, 100, 100 }, { 1, 5 }, 3, 3, 0.0625 } },
        roi_pool_test_params_float{ prop_kind::forward_inference,
        engine::kind::cpu, memory::format::nChw8c, memory::format::nc,
            memory::format::nChw8c, { { 1, 256, 100, 100 }, { 1, 5 }, 10, 10, 0.0625 } },
        roi_pool_test_params_float{ prop_kind::forward_inference,
        engine::kind::cpu, memory::format::nChw8c, memory::format::nc,
            memory::format::nChw8c, { { 1, 256, 14, 14 }, { 150, 5 }, 6, 6, 0.0625 } }
        ));
}
