////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2017 Intel Corporation. All Rights Reserved.
//
//

#ifndef MEMORY_TYPES_HPP
#define MEMORY_TYPES_HPP

#include "util/md_size.hpp"
#include "util/md_span.hpp"
#include "util/md_view.hpp"

namespace ade
{
namespace memory
{
static const constexpr std::size_t MaxDimensions = 6;
using DynMdSize = util::DynMdSize<MaxDimensions>;
using DynMdSpan = util::DynMdSpan<MaxDimensions>;

template<typename T>
using DynMdView = util::DynMdView<MaxDimensions, T>;

}
}

#endif // MEMORY_TYPES_HPP
