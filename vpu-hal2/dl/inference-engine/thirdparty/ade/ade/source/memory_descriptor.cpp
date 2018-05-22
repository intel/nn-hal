////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2017 Intel Corporation. All Rights Reserved.
//
//

#include "memory/memory_descriptor.hpp"

#include <algorithm>

#include <util/zip_range.hpp>

namespace ade
{

MemoryDescriptor::MemoryDescriptor(size_t element_size,
                                   const memory::DynMdSize& dims):
    m_elementSize(element_size),
    m_dims(dims)
{
    ASSERT(dims.dims_count() > 0);
    ASSERT(element_size > 0);
}

MemoryDescriptor::~MemoryDescriptor()
{
}

void MemoryDescriptor::addListener(IMemoryAccessListener* listener)
{
    m_accessor.addListener(listener);
}

void MemoryDescriptor::removeListener(IMemoryAccessListener* listener)
{
    m_accessor.removeListener(listener);
}

MemoryDescriptor::AccessHandle MemoryDescriptor::access(const memory::DynMdSpan& span,
                                                        MemoryAccessType accessType)
{
    return m_accessor.access(*this, span, accessType);
}

void MemoryDescriptor::commit(MemoryDescriptor::AccessHandle handle)
{
    m_accessor.commit(handle);
}

const memory::DynMdSize& MemoryDescriptor::dimensions() const
{
    return m_dims;
}

std::size_t MemoryDescriptor::elementSize() const
{
    return m_elementSize;
}

void MemoryDescriptor::setExternalView(const memory::DynMdView<void>& view)
{
    ASSERT(view.elementSize() == m_elementSize);
    ASSERT(view.count() == m_dims.dims_count());
    m_externalView = view;
    m_accessor.setNewView(view);
}

memory::DynMdView<void> MemoryDescriptor::getExternalView() const
{
    return m_externalView;
}

}
