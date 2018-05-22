////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2017 Intel Corporation. All Rights Reserved.
//
//

#ifndef TOPOLOGICAL_SORT_HPP
#define TOPOLOGICAL_SORT_HPP

#include <vector>
#include "utility"

#include "node.hpp"

#include "typed_graph.hpp"
#include "passes/pass_base.hpp"

#include "util/range.hpp"
#include "util/filter_range.hpp"

namespace ade
{
namespace passes
{

struct TopologicalSortData final
{
    struct NodesFilter final
    {
        bool operator()(const ade::NodeHandle& node) const
        {
            return nullptr != node;
        }
    };

    using NodesList = std::vector<NodeHandle>;
    using NodesRange = util::FilterRange<util::IterRange<NodesList::const_iterator>, NodesFilter>;

    TopologicalSortData(const NodesList& nodes):
        m_nodes(nodes) {}

    TopologicalSortData(NodesList&& nodes):
        m_nodes(std::move(nodes)) {}

    NodesRange nodes() const
    {
        return util::filter<NodesFilter>(util::toRange(m_nodes));
    }

    static const char* name();

private:
    NodesList m_nodes;
};

struct TopologicalSort final
{
    void operator()(TypedPassContext<TopologicalSortData> context) const;
    static const char* name();
};

struct LazyTopologicalSortChecker final
{
    bool nodeCreated(const Graph& graph, const NodeHandle& node);
    bool nodeAboutToBeDestroyed(const Graph& graph, const NodeHandle& node);

    bool edgeCreated(const Graph&, const EdgeHandle& edge);
    bool edgeAboutToBeDestroyed(const Graph& graph, const EdgeHandle& edge);
    bool edgeAboutToBeRelinked(const Graph& graph,
                               const EdgeHandle& edge,
                               const NodeHandle& newSrcNode,
                               const NodeHandle& newDstNode);
};

}
}

#endif // TOPOLOGICAL_SORT_HPP
