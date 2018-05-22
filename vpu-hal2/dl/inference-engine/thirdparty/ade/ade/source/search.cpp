////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2017 Intel Corporation. All Rights Reserved.
//
//

#include <helpers/search.hpp>

namespace ade
{
namespace
{
template<typename Visitor, typename Traverse>
void dfsHelper(const NodeHandle& node,
                  Visitor&& visitor,
                  Traverse&& direction)
{
    direction(*node, [&](const NodeHandle& nextNode)
    {
        assert(nullptr != nextNode);
        if (visitor(nextNode))
        {
            dfsHelper(nextNode,
                      std::forward<Visitor>(visitor),
                      std::forward<Traverse>(direction));
        }
    });
}
}

void dfs(const NodeHandle& node,
         util::func_ref<bool (const NodeHandle&)> visitor,
         traverse::traverse_func_type direction)
{
    ASSERT(nullptr != node);
    dfsHelper(node, visitor, direction);
}

namespace details
{

namespace
{
template<typename Traverse>
void TransitiveClosureHelperImpl(
        details::TransitiveClosureHelper::CacheT& cache,
        const NodeHandle& node,
        Traverse&& direction)
{
    ASSERT(nullptr != node);
    ASSERT(!util::contains(cache, node));
    auto& elem = cache[node];
    (void)elem; // Silence klocwork warning
    direction(*node, [&](const NodeHandle& outNode)
    {
        ASSERT(nullptr != outNode);
        if (!util::contains(cache, outNode))
        {
            TransitiveClosureHelperImpl(cache,
                                        outNode,
                                        std::forward<Traverse>(direction));
        }
        ASSERT(util::contains(cache, outNode));
        elem.insert(outNode);
        auto& nextNodes = cache[outNode];
        elem.insert(nextNodes.begin(), nextNodes.end());
    });
}
}

void details::TransitiveClosureHelper::operator()(
        details::TransitiveClosureHelper::CacheT& cache,
        const NodeHandle& node,
        traverse::traverse_func_type direction) const
{
    TransitiveClosureHelperImpl(cache, node, direction);
}

}

}
