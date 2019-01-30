/*
 * INTEL CONFIDENTIAL
 * Copyright 2017 Intel Corporation.
 *
 * The source code contained or described herein and all documents
 * related to the source code ("Material") are owned by Intel Corporation
 * or its suppliers or licensors. Title to the Material remains with
 * Intel Corporation or its suppliers and licensors. The Material may
 * contain trade secrets and proprietary and confidential information
 * of Intel Corporation and its suppliers and licensors, and is protected
 * by worldwide copyright and trade secret laws and treaty provisions.
 * No part of the Material may be used, copied, reproduced, modified,
 * published, uploaded, posted, transmitted, distributed, or disclosed
 * in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other
 * intellectual property right is granted to or conferred upon you by
 * disclosure or delivery of the Materials, either expressly, by implication,
 * inducement, estoppel or otherwise. Any license under such intellectual
 * property rights must be express and approved by Intel in writing.
 *
 * Include any supplier copyright notices as supplier requires Intel to use.
 *
 * Include supplier trademarks or logos as supplier requires Intel to use,
 * preceded by an asterisk. An asterisked footnote can be added as follows:
 * *Third Party trademarks are the property of their respective owners.
 *
 * Unless otherwise agreed by Intel in writing, you may not remove or alter
 * this notice or any other notice embedded in Materials by Intel or Intel's
 * suppliers or licensors in any way.
 */

#pragma once
#include "IRLayer.h"
#include "ie_icnn_network.hpp"
#include "ie_common.h"


class InternalNetworkImpl;

namespace IRBuilder
{

class IRDocument
{
private:
    struct Edge
    {
        struct port
        {
            int lid, pid;
        } from, to;
    };

    InternalNetworkImpl *network;
    std::vector<IRLayer> _layers; // ordered by input propagation
    std::vector<Edge> _edges;
    std::string _name;
    size_t _layer_id_cnt = 1;
    bool _processed = false;

    std::map<const float *, size_t> _segmentsMap; //org

    //std::map<const short*, size_t> _segmentsMap;

    static bool shouldRemove(const IRLayer &l);
    void process(const IRLayer &value);
    void optimize();
    void build();

    // saving functions
    static void saveOutputToIR(pugi::xml_node &parent, const InferenceEngine::DataPtr &port);
    static void saveInputToIR(pugi::xml_node &parent, int index, const InferenceEngine::DataPtr &port);

    // save a layer
    void saveToIR(std::ostream &binFile, pugi::xml_node &parent, const IRLayer &irLayer);
    // svae a blob
    void saveBlobToIR(std::ostream &binFile, const InferenceEngine::Blob::Ptr &blob, pugi::xml_node &layer, const std::string &name);

    void saveLayerToDot(std::ostream &dot, const IRLayer &irLayer) const;
    IRDocument(IRDocument &) = delete;
    IRDocument operator=(IRDocument &) = delete;

public:

    explicit IRDocument(const std::string &cs);
    ~IRDocument();

    void add(const IRLayer &ir_layer);

    // Products
    void save(std::ostream &xml_os, std::ostream &bin_os);
    void save(const std::string &filebase);
    void crateDotFile(std::ostream &dot) const;

    DelayObj createDelay(const std::string &id, const TensorDims &dims);

    InferenceEngine::InputInfo::Ptr createInput(const std::string &name, const TensorDims &dims) const;
    InferenceEngine::ICNNNetwork *buildNetwork();

    void addOutput(const IRLayer &src, int outIndx = 0);
    void addOutput(const InferenceEngine::DataPtr &src);
    void setName(const char *name);
    InferenceEngine::ICNNNetwork *getNetwork();
};

}  // namespace IRBuilder
