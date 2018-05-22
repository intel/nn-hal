//
// INTEL CONFIDENTIAL
// Copyright 2016 Intel Corporation.
//
// The source code contained or described herein and all documents
// related to the source code ("Material") are owned by Intel Corporation
// or its suppliers or licensors. Title to the Material remains with
// Intel Corporation or its suppliers and licensors. The Material may
// contain trade secrets and proprietary and confidential information
// of Intel Corporation and its suppliers and licensors, and is protected
// by worldwide copyright and trade secret laws and treaty provisions.
// No part of the Material may be used, copied, reproduced, modified,
// published, uploaded, posted, transmitted, distributed, or disclosed
// in any way without Intel's prior express written permission.
//
// No license under any patent, copyright, trade secret or other
// intellectual property right is granted to or conferred upon you by
// disclosure or delivery of the Materials, either expressly, by implication,
// inducement, estoppel or otherwise. Any license under such intellectual
// property rights must be express and approved by Intel in writing.
//
// Include any supplier copyright notices as supplier requires Intel to use.
//
// Include supplier trademarks or logos as supplier requires Intel to use,
// preceded by an asterisk. An asterisked footnote can be added as follows:
// *Third Party trademarks are the property of their respective owners.
//
// Unless otherwise agreed by Intel in writing, you may not remove or alter
// this notice or any other notice embedded in Materials by Intel or Intel's
// suppliers or licensors in any way.
//
#pragma once

#include <string>
#include <map>
#include "cnn_network_impl.hpp"
#include "ie_layers.h"
#include "parsers.h"
#include <vector>

namespace InferenceEngine {
namespace details {
struct WeightSegment {
    Precision precision;
    // offset in bytes of the global weights array
    size_t start = 0;
    // size in bytes
    size_t size = 0;

    inline size_t getEnd() const { return start + size; }

    // checks if this segment is in the range of 0 to rangeSize, safer than using getEnd() to avoid int overflow
    inline bool inRange(size_t rangeSize) const {
        return start < rangeSize && (rangeSize - start) >= size;
    }
};

struct LayerParseParameters {
    struct LayerPortData {
        int           portId;
        Precision     precision;
        SizeVector    dims;
    };
    InferenceEngine::LayerParams prms;
    int layerId = -1;
    std::vector<LayerPortData> inputPorts;
    std::vector<LayerPortData> outputPorts;
    std::map<std::string, WeightSegment> blobs;

    void addOutputPort(const LayerPortData &port);
    void addInputPort(const LayerPortData &port);
};

class BaseCreator {
    std::string type_;
protected:
    explicit BaseCreator(const std::string& type) : type_(type) {}

    virtual ~BaseCreator() {}

public:
    static int version_;

    virtual InferenceEngine::CNNLayer* CreateLayer(pugi::xml_node& node, LayerParseParameters& layerParsePrms) = 0;

    bool shouldCreate(const std::string& nodeType) const { return nodeType.compare(type_) == 0; }
};

class V2FormatParser : public IFormatParser {
public:
    explicit V2FormatParser(int version);

    CNNNetworkImplPtr Parse(pugi::xml_node& root) override;

    Blob::Ptr GetBlobFromSegment(const TBlob<uint8_t>::Ptr& weights, const WeightSegment & weight_segment) const;
    void SetWeights(const TBlob<uint8_t>::Ptr& weights) override;
    void ParseDims(SizeVector& dims, const pugi::xml_node &node) const;

private:
    int _version;
    Precision _defPrecision;
    std::map<std::string, LayerParseParameters> layersParseInfo;
    std::map<std::string, DataPtr> _portsToData;

    CNNNetworkImplPtr _network;
    std::map<std::string, std::vector<WeightSegment>> _preProcessSegments;
    std::vector<BaseCreator*> getCreators() const;
    void ParsePort(LayerParseParameters::LayerPortData& port, pugi::xml_node &node) const;
    void ParseGenericParams(pugi::xml_node& node, LayerParseParameters& layerParsePrms) const;
    CNNLayer* CreateLayer(pugi::xml_node& node, LayerParseParameters& prms) const;

    void SetLayerInput(CNNNetworkImpl& network, const std::string& data, CNNLayerPtr& targetLayer, int inputPort);

    DataPtr ParseInputData(pugi::xml_node& root) const;

    void ParsePreProcess(pugi::xml_node& node);
};
}  // namespace details
}  // namespace InferenceEngine
