////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2017 Intel Corporation. All Rights Reserved.
//
//

#ifndef PASS_BASE_HPP
#define PASS_BASE_HPP

namespace ade
{

class Graph;

template<typename...>
class TypedGraph;

namespace passes
{
struct PassContext
{
    Graph& graph;
};

template<typename... Types>
struct TypedPassContext
{
    TypedGraph<Types...> graph;

    TypedPassContext(PassContext& context):
        graph(context.graph) {}
};

}

}


#endif // PASS_BASE_HPP
