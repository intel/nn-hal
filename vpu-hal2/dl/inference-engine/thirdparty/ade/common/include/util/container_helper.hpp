///////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2016-2017 Intel Corporation. All Rights Reserved.
//

#ifndef UTIL_CONTAINER_HELPER_HPP
#define UTIL_CONTAINER_HELPER_HPP

#include <vector>
#include <array>

#include "util/memory_range.hpp"

namespace util
{

template<typename T>
inline auto data(const std::vector<T>& vector) -> decltype(vector.data())
{
    return vector.data();
}

template<typename T>
inline std::size_t size(const std::vector<T>& vector)
{
    return vector.size();
}

template<typename T>
inline auto slice(const std::vector<T>& vector, const std::size_t start, const std::size_t newSize)
->decltype(memory_range(data(vector), size(vector)).Slice(start, newSize))
{
    return memory_range(data(vector), size(vector)).Slice(start, newSize);
}

template<typename T, std::size_t Size>
inline auto data(const std::array<T, Size>& arr) -> decltype(arr.data())
{
    return arr.data();
}

template<typename T, std::size_t Size>
inline std::size_t size(const std::array<T, Size>& arr)
{
    return arr.size();
}

template<typename T, std::size_t Size>
inline auto slice(const std::array<T, Size>& arr, const std::size_t start, const std::size_t newSize)
->decltype(memory_range(data(arr), size(arr)).Slice(start, newSize))
{
    return memory_range(data(arr), size(arr)).Slice(start, newSize);
}

}

#endif
