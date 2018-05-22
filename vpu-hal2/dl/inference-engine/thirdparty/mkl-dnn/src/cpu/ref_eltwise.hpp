/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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

#ifndef CPU_REF_ELTWISE_HPP
#define CPU_REF_ELTWISE_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_eltwise_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct ref_eltwise_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_eltwise_fwd_pd_t {
        pd_t(engine_t *engine, const eltwise_desc_t *adesc,
                const primitive_attr_t *attr,
                const eltwise_fwd_pd_t *hint_fwd_pd)
            : cpu_eltwise_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , is_dense(false) {}

        DECLARE_COMMON_PD_T("ref:any", ref_eltwise_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(engine()->kind() == engine_kind::cpu);

            is_dense = memory_desc_wrapper(src_pd()).is_dense();
            bool ok = true
                && utils::one_of(desc()->prop_kind, forward_training,
                        forward_inference)
                && utils::everyone_is(data_type, desc()->data_desc.data_type)
                && utils::implication(!is_dense, src_pd()->desc()->ndims == 4)
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return status::success;
        }

        bool is_dense;
    };

    ref_eltwise_fwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) {
        if (conf_.is_dense) execute_forward_dense();
        else execute_forward_generic();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward_dense();
    void execute_forward_generic();
    pd_t conf_;
};

template <impl::data_type_t data_type>
struct ref_eltwise_bwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_eltwise_bwd_pd_t {
        pd_t(engine_t *engine, const eltwise_desc_t *adesc,
                const primitive_attr_t *attr,
                const eltwise_fwd_pd_t *hint_fwd_pd)
            : cpu_eltwise_bwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , is_dense_(false) {}

        DECLARE_COMMON_PD_T("ref:any", ref_eltwise_bwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true && desc()->prop_kind == backward_data
                    && utils::everyone_is(data_type,
                               desc()->data_desc.data_type,
                               desc()->diff_data_desc.data_type)
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            const bool same_fmt = memory_desc_wrapper(diff_dst_pd())
                == memory_desc_wrapper(src_pd());
            is_dense_ = memory_desc_wrapper(src_pd()).is_dense() && same_fmt;

            if (!utils::implication(!is_dense_, src_pd()->desc()->ndims == 4))
                return status::unimplemented;

            return status::success;
        }

        bool is_dense_;
    };

    ref_eltwise_bwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) {
        if (conf_.is_dense_) execute_backward_dense();
        else execute_backward_generic();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_dense();
    void execute_backward_generic();
    pd_t conf_;
};

template <typename T, typename A> inline T relu_fwd(T s, A alpha) {
    return s > 0 ? s : (T)(s * alpha);
}
template <typename T, typename A> inline T relu_bwd(T dd, T s, A alpha) {
    return s > 0 ? dd : (T)(dd * alpha);
}

template <typename T> T tanh_fwd(T s) {
    const float e = ::expf((float)(2 * s)); /* maybe replace with -2*s? */
    return (T)((e - 1) / (e + 1));
}
template <typename T> T tanh_bwd(T dd, T s) {
    const float e = ::expf((float)(2 * s)); /* maybe replace with -2*s? */
    const float th = (e - 1.f) / (e + 1.f);
    return (T)(dd * (1 - th * th));
}

template <typename T, typename A> T elu_fwd(T s, A alpha) {
    return s > 0 ? s : (T)(alpha * (::expf((float)s) - 1.f));
}
template <typename T, typename A> T elu_bwd(T dd, T s, A alpha) {
    return (T)(dd * (s > 0 ? 1 : alpha * ::expf((float)s)));
}

template <typename T>
T square_fwd(T s) {
    return s * s;
}

template <typename T>
T square_bwd(T dd, T s) {
    return dd * 2*s;
}

template <typename T>
T abs_fwd(T s) {
    return s > 0 ? s : -s;
}

template <typename T>
T abs_bwd(T dd, T s) {
    return s > 0 ? dd : s < 0 ? -dd : 0;
}

template <typename T>
T sqrt_fwd(T s) {
    return s > 0 ? (T)(::sqrtf((float)(s))) : 0;
}

template <typename T>
T sqrt_bwd(T dd, T s) {
    return s > 0
        ? (T)(dd / (2 * ::sqrtf((float)(s))))
        : 0;
}

template <typename T, typename A>
T linear_fwd(T s, A alpha, A beta) {
    return (T)(alpha * s + beta);
}

template <typename T, typename A>
T linear_bwd(T dd, T s, A alpha, A beta) {
    (void) s;
    (void) beta;
    return (T)(dd * alpha);
}

template <typename T, typename A>
T bounded_relu_fwd(T s, A alpha) {
    s = s > 0 ? s : 0;
    return s > alpha ? (T)(alpha) : s;
}

template <typename T, typename A>
T bounded_relu_bwd(T dd, T s, A alpha) {
    return dd * (0 < s && s < alpha ? 1 : 0);
}

template <typename T>
T soft_relu_fwd(T s) {
    return (T)(::logf(1 + ::expf((float)s)));
}

template <typename T>
T soft_relu_bwd(T dd, T s) {
    return (T)(dd / (1 + ::expf((float)(-s))));
}

template <typename T>
T logistic_fwd(T s) {
    T v = (T)(::expf((float)s));
    return v / (v + 1);
}

template <typename T>
T logistic_bwd(T dd, T s) {
    T v = (T)(::expf((float)(-s)));
    return dd * v / ((v + 1) * (v + 1));
}

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
