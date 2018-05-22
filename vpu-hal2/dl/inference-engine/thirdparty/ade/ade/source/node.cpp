////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2006-2016 Intel Corporation. All Rights Reserved.
//
//

#include "node.hpp"

#include <memory>

#include "util/assert.hpp"
#include "util/algorithm.hpp"

#include "graph.hpp"
#include "edge.hpp"

namespace ade
{

Node::Node(Graph* parent):
    m_parent(parent)
{

}

Node::~Node()
{
    unlink();
}

void Node::unlink()
{
    ASSERT(nullptr != m_parent);
    for (auto& edge: m_inEdges) //TODO: join ranges
    {
        ASSERT(nullptr != edge);
        ASSERT(this == edge->m_nextNode);
        edge->m_nextNode = nullptr;
        m_parent->removeEdge(edge);
    }
    m_inEdges.clear();

    for (auto& edge: m_outEdges) //TODO: join ranges
    {
        ASSERT(nullptr != edge);
        ASSERT(this == edge->m_prevNode);
        edge->m_prevNode = nullptr;
        m_parent->removeEdge(edge);
    }
    m_outEdges.clear();
}

void Node::addInEdge(Edge* edge)
{
    ASSERT(nullptr != edge);
    ASSERT(m_inEdges.end() == util::find(m_inEdges, edge));
    m_inEdges.emplace_back(edge);
}

void Node::removeInEdge(Edge* edge)
{
    ASSERT(nullptr != edge);
    // Nodes usually have only a small amount of connections so linear search should be fine here
    auto it = util::find(m_inEdges, edge);
    ASSERT(m_inEdges.end() != it);
    util::unstable_erase(m_inEdges, it);
}

void Node::addOutEdge(Edge* edge)
{
    ASSERT(nullptr != edge);
    ASSERT(m_outEdges.end() == util::find(m_outEdges, edge));
    m_outEdges.emplace_back(edge);
}

void Node::removeOutEdge(Edge* edge)
{
    ASSERT(nullptr != edge);
    // Nodes usually have only a small amount of connections so linear search should be fine here
    auto it = util::find(m_outEdges, edge);
    ASSERT(m_outEdges.end() != it);
    util::unstable_erase(m_outEdges, it);
}

Graph* Node::getParent() const
{
    return m_parent;
}

Node::EdgeSetRange Node::inEdges()
{
    return util::map<HandleMapper>(util::toRange(m_inEdges));
}

Node::EdgeSetCRange Node::inEdges() const
{
    return util::map<HandleMapper>(util::toRange(m_inEdges));
}

Node::EdgeSetRange Node::outEdges()
{
    return util::map<HandleMapper>(util::toRange(m_outEdges));
}

Node::EdgeSetCRange Node::outEdges() const
{
    return util::map<HandleMapper>(util::toRange(m_outEdges));
}

Node::InNodeSetRange Node::inNodes()
{
    return util::map<InEdgeMapper>(inEdges());
}

Node::InNodeSetCRange Node::inNodes() const
{
    return util::map<InEdgeMapper>(inEdges());
}

Node::OutNodeSetRange Node::outNodes()
{
    return util::map<OutEdgeMapper>(outEdges());
}

Node::OutNodeSetCRange Node::outNodes() const
{
    return util::map<OutEdgeMapper>(outEdges());
}

EdgeHandle Node::HandleMapper::operator()(Edge* obj) const
{
    ASSERT(nullptr != obj);
    return Graph::HandleMapper()(obj);
}

NodeHandle Node::InEdgeMapper::operator()(const EdgeHandle& handle) const
{
    ASSERT(nullptr != handle);
    return handle->srcNode();
}

NodeHandle Node::OutEdgeMapper::operator()(const EdgeHandle& handle) const
{
    ASSERT(nullptr != handle);
    return handle->dstNode();
}

}
