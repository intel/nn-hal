////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2006-2017 Intel Corporation. All Rights Reserved.
//
//

#ifndef UTIL_ZIP_RANGE_HPP
#define UTIL_ZIP_RANGE_HPP

#include "util/tuple.hpp"
#include "util/range.hpp"
#include "util/assert.hpp"
#include "util/iota_range.hpp"
#include "util/range_iterator.hpp"

namespace util
{

inline namespace Range
{

template<typename... Ranges>
struct ZipRange : public IterableRange<ZipRange<Ranges...>>
{
    using tuple_t = decltype(std::make_tuple(toRange(std::declval<Ranges>())...));
    tuple_t ranges;

    ZipRange() = default;
    ZipRange(const ZipRange&) = default;
    ZipRange(ZipRange&&) = default;
    ZipRange& operator=(const ZipRange&) = default;
    ZipRange& operator=(ZipRange&&) = default;
    ZipRange(Ranges&& ...r): ranges{toRange(std::forward<Ranges>(r))...} {}

    template<int... S>
    auto unpackTuple(details::Seq<S...>) ->
    decltype(tuple_remove_rvalue_refs(std::get<S>(ranges).front()...))
    {
        return tuple_remove_rvalue_refs(std::get<S>(ranges).front()...);
    }

    template<int... S>
    auto unpackTuple(details::Seq<S...>) const ->
    decltype(tuple_remove_rvalue_refs(std::get<S>(ranges).front()...))
    {
        return tuple_remove_rvalue_refs(std::get<S>(ranges).front()...);
    }

    bool empty() const
    {
        ::util::details::RangeChecker checker;
        tupleForeach(ranges, checker);
        return checker.empty;
    }

    void popFront()
    {
        ASSERT(!empty());
        tupleForeach(ranges, ::util::details::RangeIncrementer());
    }

    auto front()->decltype(this->unpackTuple(details::gen_t<sizeof...(Ranges)>{}))
    {
        ASSERT(!empty());
        return unpackTuple(details::gen_t<sizeof...(Ranges)>{});
    }

    auto front() const->decltype(this->unpackTuple(details::gen_t<sizeof...(Ranges)>{}))
    {
        ASSERT(!empty());
        return unpackTuple(details::gen_t<sizeof...(Ranges)>{});
    }


};

template<typename... Ranges>
inline ZipRange<Ranges...> zip(Ranges&&... ranges)
{
    return {std::forward<Ranges>(ranges)...};
}

template<typename... Containers>
inline auto indexed(Containers&&... conts) ->
decltype(zip(iota<std::size_t>(), std::forward<Containers>(conts)...))
{
    return zip(iota<std::size_t>(), std::forward<Containers>(conts)...);
}

}
}

#endif // UTIL_ZIP_RANGE_HPP
