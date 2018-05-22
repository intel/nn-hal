////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2006-2016 Intel Corporation. All Rights Reserved.
//
//

#ifndef GRAPH_LISTENER_HPP
#define GRAPH_LISTENER_HPP

#include "handle.hpp"

namespace ade
{

class Graph;
class Node;
class Edge;
using EdgeHandle = Handle<Edge>;
using NodeHandle = Handle<Node>;

class IGraphListener
{
public:
    virtual ~IGraphListener() = default;

    virtual void nodeCreated(const Graph& graph, const NodeHandle& node) = 0;
    virtual void nodeAboutToBeDestroyed(const Graph& graph, const NodeHandle& node) = 0;

    virtual void edgeCreated(const Graph&, const EdgeHandle& edge) = 0;
    virtual void edgeAboutToBeDestroyed(const Graph& graph, const EdgeHandle& edge) = 0;
    virtual void edgeAboutToBeRelinked(const Graph& graph,
                                       const EdgeHandle& edge,
                                       const NodeHandle& newSrcNode,
                                       const NodeHandle& newDstNode) = 0;
};

}

#endif // GRAPH_LISTENER_HPP
