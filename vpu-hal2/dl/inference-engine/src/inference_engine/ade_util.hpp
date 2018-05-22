////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2017 Intel Corporation. All Rights Reserved.
//
//

#pragma once

#include <ie_layers.h>

namespace ade {
class Graph;
}  // namespace ade

namespace InferenceEngine {

struct CNNLayerMetadata {
    CNNLayerPtr layer;

    static const char* name();
};

class ICNNNetwork;
void translateNetworkToAde(ade::Graph& gr, ICNNNetwork& netrwork);
}  // namespace InferenceEngine

