////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2017 Intel Corporation. All Rights Reserved.
//
//

#ifndef UTIL_FUNC_REF_HPP
#define UTIL_FUNC_REF_HPP

#include <cstdint>
#include <utility>

#include "util/type_traits.hpp"
#include "util/assert.hpp"

namespace util
{
template<typename>
class func_ref; // undefined

/// Non-owning callable wrapper
template<typename R, typename... Args>
class func_ref<R(Args...)>
{
    using func_t = R(*)(uintptr_t, Args...);
    uintptr_t m_context = 0;
    func_t m_func = nullptr;

    template<typename T>
    static R thunk(uintptr_t context, Args... args)
    {
        T* obj = reinterpret_cast<T*>(context);
        return (*obj)(std::forward<Args>(args)...);
    }

public:
    template<typename Callable>
    func_ref(Callable&& callable):
        m_context(reinterpret_cast<uintptr_t>(&callable)),
        m_func(&thunk<util::remove_reference_t<Callable>>)
    {
        using actual_result_type = util::result_of_t<Callable(Args...)>;

        // If this condition doesn't hold, then thunk will return a reference
        // to the temporary returned by callable.
        static_assert(
            !std::is_reference<R>::value || std::is_reference<actual_result_type>::value,
            "If R is a reference, callable must also return a reference");
    }

    R operator()(Args... args) const
    {
        ASSERT(0 != m_context);
        ASSERT(nullptr != m_func);
        return m_func(m_context, std::forward<Args>(args)...);
    }
};
}

#endif // UTIL_FUNC_REF_HPP
