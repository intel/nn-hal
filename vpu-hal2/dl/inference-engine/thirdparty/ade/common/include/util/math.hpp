////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2017 Intel Corporation. All Rights Reserved.
//
//

#ifndef UTIL_MATH_HPP
#define UTIL_MATH_HPP

#include <type_traits>

#include "util/assert.hpp"

namespace util
{
template<typename T>
inline auto is_pow2(T val)
->typename std::enable_if<std::is_integral<T>::value, bool>::type
{
    return (val & (val - 1)) == 0;
}

template<typename T>
inline auto align_size(T size, T align)
->typename std::enable_if<std::is_integral<T>::value, T>::type
{
    ASSERT(size > 0);
    ASSERT(align > 0);
    ASSERT(is_pow2(align));
    return (size + (align - 1)) & ~(align - 1);
}

}

#endif // UTIL_MATH_HPP
