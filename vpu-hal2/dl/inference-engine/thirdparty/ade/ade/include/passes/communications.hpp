////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2017 Intel Corporation. All Rights Reserved.
//
//

#ifndef COMMUNICATIONS_HPP
#define COMMUNICATIONS_HPP

#include <passes/pass_base.hpp>

#include <metatypes/metatypes.hpp>

namespace ade
{
namespace passes
{

struct ConnectCommChannels final
{
    using Context = ade::passes::TypedPassContext<ade::meta::CommNode,
                                                  ade::meta::DataObject,
                                                  ade::meta::CommChannel,
                                                  ade::meta::NodeInfo,
                                                  ade::meta::CommConsumerCallback,
                                                  ade::meta::CommProducerCallback,
                                                  ade::meta::Finalizers>;
    void operator()(Context ctx) const;
    static const char* name();
};

}
}

#endif // COMMUNICATIONS_HPP
