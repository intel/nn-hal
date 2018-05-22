////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2017 Intel Corporation. All Rights Reserved.
//
//

#ifndef UTIL_MD_CAST_HPP
#define UTIL_MD_CAST_HPP

namespace util
{
// TODO: find a proper place for this
constexpr static const std::size_t MaxDimensions = 6;

namespace detail
{
template<typename Target>
struct md_cast_helper; // Undefined
}

template<typename Dst, typename Src>
Dst md_cast(const Src& src)
{
    return detail::md_cast_helper<Dst>(src);
}
}

#endif // UTIL_MD_CAST_HPP
