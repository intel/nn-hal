////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2006-2016 Intel Corporation. All Rights Reserved.
//
//

#ifndef EDGE_HPP
#define EDGE_HPP

#include <memory>
#include <string>

#include "handle.hpp"
#include "metadata.hpp"

namespace ade
{

class Graph;
class Node;
class Edge;
using EdgeHandle = Handle<Edge>;
using NodeHandle = Handle<Node>;

class Edge final : public std::enable_shared_from_this<Edge>
{
public:
    NodeHandle srcNode() const;
    NodeHandle dstNode() const;
private:
    friend class Graph;
    friend class Node;

    Edge(Node* prev, Node* next);
    ~Edge();
    Edge(const Edge&) = delete;
    Edge& operator=(const Edge&) = delete;

    Graph* getParent() const;

    void unlink();
    void resetPrevNode(Node* newNode);
    void resetNextNode(Node* newNode);

    Node* m_prevNode = nullptr;
    Node* m_nextNode = nullptr;
};

}

#endif // EDGE_HPP
