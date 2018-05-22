////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2017 Intel Corporation. All Rights Reserved.
//
//

#ifndef HASH_HPP
#define HASH_HPP

#include <cstddef> //size_t

namespace util
{
inline std::size_t hash_combine(std::size_t seed, std::size_t val)
{
    // Hash combine formula from boost
    return seed ^ (val + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}
} // namespace util

#endif // HASH_HPP
