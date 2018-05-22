////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2006-2016 Intel Corporation. All Rights Reserved.
//
//

#ifndef METADATA_HPP
#define METADATA_HPP

#include <unordered_map>
#include <utility>

#include "util/assert.hpp"
#include "util/any.hpp"
#include "util/range.hpp"

namespace ade
{

class Graph;
class Metadata;

class MetadataId final
{
    friend class Graph;
    friend class Metadata;

    MetadataId(void* id);

    void* m_id = nullptr;
public:
    MetadataId() = default;
    MetadataId(std::nullptr_t) {}
    MetadataId(const MetadataId&) = default;
    MetadataId& operator=(const MetadataId&) = default;
    MetadataId& operator=(std::nullptr_t) { m_id = nullptr; return *this; }

    bool operator==(const MetadataId& other) const;
    bool operator!=(const MetadataId& other) const;

    bool isNull() const;
};

bool operator==(std::nullptr_t, const MetadataId& other);
bool operator==(const MetadataId& other, std::nullptr_t);
bool operator!=(std::nullptr_t, const MetadataId& other);
bool operator!=(const MetadataId& other, std::nullptr_t);

class Metadata final
{
    struct IdHash final
    {
        std::size_t operator()(const MetadataId& id) const;
    };
public:
    using MetadataStore = std::unordered_map<MetadataId, util::any, IdHash>;
    using MetadataRange  = util::IterRange<MetadataStore::iterator>;
    using MetadataCRange = util::IterRange<MetadataStore::const_iterator>;

    Metadata();

    util::any& operator[](const MetadataId& id);

    bool contains(const MetadataId& id) const;
    void erase(const MetadataId& id);

    template<typename T>
    void set(const MetadataId& id, T&& val)
    {
        ASSERT(nullptr != id);
        m_data.erase(id);
        m_data.emplace(id, std::forward<T>(val));
    }

    template<typename T>
    T& get(const MetadataId& id)
    {
        ASSERT(nullptr != id);
        ASSERT(contains(id));
        auto ret = util::any_cast<T>(&(m_data.find(id)->second));
        ASSERT(nullptr != ret);
        return *ret;
    }

    template<typename T>
    const T& get(const MetadataId& id) const
    {
        ASSERT(nullptr != id);
        ASSERT(contains(id));
        auto ret = util::any_cast<T>(&(m_data.find(id)->second));
        ASSERT(nullptr != ret);
        return *ret;
    }

    template<typename T>
    T get(const MetadataId& id, T&& def) const
    {
        ASSERT(nullptr != id);
        auto it = m_data.find(id);
        if (m_data.end() == it)
        {
            return std::forward<T>(def);
        }
        auto ret = util::any_cast<T>(&(it->second));
        ASSERT(nullptr != ret);
        return *ret;
    }

    MetadataRange  all();
    MetadataCRange all() const;

    /// This operation rarely needed so do it explicitly
    void copyFrom(const Metadata& data);

private:
    Metadata(const Metadata&) = delete;
    Metadata& operator=(const Metadata&) = delete;

    MetadataStore m_data;
};

}

#endif // METADATA_HPP
