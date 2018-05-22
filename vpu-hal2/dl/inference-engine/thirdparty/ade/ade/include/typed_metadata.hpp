////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2017 Intel Corporation. All Rights Reserved.
//
//

#ifndef TYPED_METADATA_HPP
#define TYPED_METADATA_HPP

#include <array>
#include <type_traits>

#include "util/algorithm.hpp"

#include "metadata.hpp"

namespace ade
{
template<bool IsConst, typename... Types>
class TypedMetadata
{
    using IdArray = std::array<ade::MetadataId, sizeof...(Types)>;
    using MetadataT = typename std::conditional<IsConst, const ade::Metadata&, ade::Metadata&>::type;
    const IdArray& m_ids;
    MetadataT m_metadata;

    template<typename T>
    ade::MetadataId getId() const
    {
        const auto index = util::type_list_index<typename std::decay<T>::type, Types...>::value;
        return m_ids[index];
    }

public:
    TypedMetadata(const IdArray& ids, MetadataT meta):
        m_ids(ids), m_metadata(meta) {}

    TypedMetadata(const TypedMetadata& other):
        m_ids(other.m_ids), m_metadata(other.m_metadata) {}

    TypedMetadata& operator=(const TypedMetadata&) = delete;

    template<bool, typename...>
    friend class TypedMetadata;

    template<typename T>
    bool contains() const
    {
        return m_metadata.contains(getId<T>());
    }

    template<typename T>
    void erase()
    {
        m_metadata.erase(getId<T>());
    }

    template<typename T>
    void set(T&& val)
    {
        m_metadata.set(getId<T>(), std::forward<T>(val));
    }

    template<typename T>
    auto get() const
    ->typename std::conditional<IsConst, const T&, T&>::type
    {
        return m_metadata.template get<T>(getId<T>());
    }

    template<typename T>
    T get(T&& def) const
    {
        return m_metadata.get(getId<T>(), std::forward<T>(def));
    }

    template<bool IsC, typename... T>
    void copyFrom(const TypedMetadata<IsC, T...>& meta)
    {
        m_metadata.copyFrom(meta.m_metadata);
    }
};

}

#endif // TYPED_METADATA_HPP
