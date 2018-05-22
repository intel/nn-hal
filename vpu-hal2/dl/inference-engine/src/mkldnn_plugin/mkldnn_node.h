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

#include <ie_api.h>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <ie_common.h>
#include <caseless.hpp>
#include "mkldnn_dims.h"
#include "mkldnn_memory.h"
#include "mkldnn_edge.h"
#include "mkldnn_descriptor.h"
#include "mkldnn/iml_type_mapper.h"
#include "mkldnn_extension_mngr.h"

namespace MKLDNNPlugin {

using MKLDNNNodePtr = std::shared_ptr<MKLDNNNode>;
using MKLDNNNodeWeakPtr = std::weak_ptr<MKLDNNNode>;

enum Type {
    Unknown,
    Generic,
    Reorder,
    Input,
    Output,
    Convolution,
    Deconvolution,
    Convolution_Sum,
    Convolution_Activation,
    Convolution_Sum_Activation,
    Clamp,
    Activation,
    Lrn,
    Pooling,
    FullyConnected,
    SoftMax,
    Split,
    Concatenation,
    Power,
    ScaleShift,
    Eltwise,
    Crop,
    Reshape,
    Tile,
    SimplerNMS,
    ROIPooling,
    BatchNormalization,
    Flatten,
    Permute,
    Copy,
    MemoryOutput,
    MemoryInput,
};

static Type TypeFromName(const std::string type) {
    static caseless_unordered_map<std::string, Type> type_to_name_tbl = {
            { "Unknown", Unknown },
            { "Input", Input },
            { "Const", Input },
            { "Convolution", Convolution },
            { "ReLU", Activation },
            { "ELU", Activation },
            { "Sigmoid", Activation },
            { "Logistic", Activation },
            { "TanH", Activation },
            { "Activation", Activation },
            { "Clamp", Clamp },
            { "Norm", Lrn },
            { "LRN", Lrn },
            { "Pooling", Pooling },
            { "FullyConnected", FullyConnected },
            { "InnerProduct", FullyConnected },
            { "Softmax", SoftMax },
            { "SoftMax", SoftMax },
            { "Split", Split },
            { "Slice", Split },
            { "Concat", Concatenation },
            { "Power", Power },
            { "Deconvolution", Deconvolution },
            { "Eltwise", Eltwise },
            { "Crop", Crop },
            { "ScaleShift", ScaleShift },
            { "Reshape", Reshape },
            { "Tile", Tile },
            { "SimplerNMS", SimplerNMS },
            { "ROIPooling", ROIPooling },
            { "BatchNormalization", BatchNormalization },
            { "Flatten", Flatten },
            { "Permute", Permute },
            { "Copy", Copy },
            { "MemoryInput", MemoryInput},  // for construction from name ctor, arbitrary name is used
            { "Memory", MemoryOutput },  // for construction from layer ctor
    };

    if (type_to_name_tbl.find(type) != type_to_name_tbl.end()) {
        return type_to_name_tbl[type];
    } else {
        return Unknown;
    }
}

class MKLDNNPrimitiveDescInfo {
public:
    MKLDNNPrimitiveDescInfo(const mkldnn::engine& eng, const std::vector<MKLDNNMemoryDesc>& in,
                            const std::vector<MKLDNNMemoryDesc>& out, impl_desc_type type)
            : engine(eng), inDescs(in), outDescs(out) {
        implementationType = type;
    }
    MKLDNNPrimitiveDescInfo(const mkldnn::engine& eng, const std::vector<MKLDNNMemoryDesc>& in,
                            const std::vector<MKLDNNMemoryDesc>& out, const char *desc_native_name)
            : engine(eng), inDescs(in), outDescs(out) {
        implementationType = parse_impl_name(desc_native_name);
    }

    MKLDNNPrimitiveDescInfo(const mkldnn::engine& eng, const std::vector<MKLDNNMemoryDesc>& in,
                            const std::vector<MKLDNNMemoryDesc>& out, const std::vector<MKLDNNMemoryDesc>& intDescs,
                            impl_desc_type type): engine(eng), inDescs(in), outDescs(out), internalDescs(intDescs) {
        implementationType = type;
    }

    MKLDNNPrimitiveDescInfo(const mkldnn::engine& eng,  const std::vector<MKLDNNMemoryDesc>& in,
                            const std::vector<MKLDNNMemoryDesc>& out, const std::vector<MKLDNNMemoryDesc>& intDescs,
                            const char *desc_native_name): engine(eng), inDescs(in), outDescs(out),
                                                           internalDescs(intDescs) {
        implementationType = parse_impl_name(desc_native_name);
    }

    MKLDNNPrimitiveDescInfo(const MKLDNNPrimitiveDescInfo &descInfo): engine(descInfo.engine),
                                                                      implementationType(descInfo.implementationType),
                                                                      inDescs(descInfo.inDescs),
                                                                      outDescs(descInfo.outDescs),
                                                                      internalDescs(descInfo.internalDescs) {}

    MKLDNNPrimitiveDescInfo(MKLDNNPrimitiveDescInfo &&descInfo): engine(descInfo.engine),
                                                                 implementationType(descInfo.implementationType),
                                                                 inDescs(descInfo.inDescs),
                                                                 outDescs(descInfo.outDescs),
                                                                 internalDescs(descInfo.internalDescs) {}

    MKLDNNPrimitiveDescInfo &operator=(const MKLDNNPrimitiveDescInfo &descInfo) {
        if (engine != descInfo.engine)
            THROW_IE_EXCEPTION << "Cannot copy descriptor info.";
        implementationType = descInfo.implementationType;
        inDescs = descInfo.inDescs;
        outDescs = descInfo.outDescs;
        internalDescs = descInfo.internalDescs;
        return *this;
    }

    const mkldnn::engine& getEngine() const {
        return engine;
    }

    const std::vector<MKLDNNMemoryDesc> getInputDescs() const {
        return inDescs;
    }
    std::vector<MKLDNNMemoryDesc>& getInputDescs() {
        return inDescs;
    }

    const std::vector<MKLDNNMemoryDesc> getOutputDescs() const {
        return outDescs;
    }
    std::vector<MKLDNNMemoryDesc>& getOutputDescs() {
        return outDescs;
    }

    const std::vector<MKLDNNMemoryDesc> getInternalDescs() const {
        return internalDescs;
    }
    std::vector<MKLDNNMemoryDesc>& getInternalDescs() {
        return internalDescs;
    }

    impl_desc_type getImplementationType() const {
        return implementationType;
    }

private:
    const mkldnn::engine engine;
    std::vector<MKLDNNMemoryDesc> inDescs;
    std::vector<MKLDNNMemoryDesc> outDescs;
    std::vector<MKLDNNMemoryDesc> internalDescs;
    impl_desc_type implementationType;
};

class MKLDNNNode : public InferenceEngine::details::no_copy {
public:
    static MKLDNNNode* CreateNode(const Type type, const std::string &name,
                                  const MKLDNNExtensionManager::Ptr& extMgr);
    static MKLDNNNode* CreateNode(const InferenceEngine::CNNLayerPtr& layer,
                                  const MKLDNNExtensionManager::Ptr& extMgr);

    virtual ~MKLDNNNode() {}

    void addEdge(const MKLDNNEdgeWeakPtr& edge, size_t pIndex, size_t cIndex);
    void removeEdge(const MKLDNNEdgeWeakPtr& edge);

    void cleanup();
    void remove();

    const std::vector<MKLDNNEdgeWeakPtr> &getParentEdges() const noexcept {
        return parentEdges;
    }

    const std::vector<MKLDNNEdgeWeakPtr> &getChildEdges() const noexcept {
        return childEdges;
    }

    const MKLDNNEdgePtr getParentEdgeAt(size_t idx) const;
    virtual const MKLDNNEdgePtr getChildEdgeAt(size_t idx) const;


    bool isDropped() {
        return (isEdgesEmpty(childEdges) && isEdgesEmpty(parentEdges));
    }

    virtual bool isConstant(bool fromCache);

    void fuseWith(MKLDNNNodePtr fuse) {
        fusedWith.push_back(fuse);
    }

    void mergeWith(MKLDNNNodePtr merge) {
        mergedWith.push_back(merge);
    }

    const std::vector <MKLDNNNodePtr> &getMergeWith() {
        return mergedWith;
    }

    const std::string getName() const {
        return name;
    }

    Type getType() const {
        return type;
    }

    const InferenceEngine::CNNLayerPtr &getCnnLayer() {
        return cnnLayer;
    }

    const std::vector<MKLDNNPrimitiveDescInfo>& getSupportedPrimitiveDescriptors() const {
        return supportedPrimitiveDescriptors;
    }

    inline const MKLDNNPrimitiveDescInfo* getSelectedPrimitiveDescriptor() const {
        if (selectedPrimitiveDescriptorIndex < 0 ||
            selectedPrimitiveDescriptorIndex >= supportedPrimitiveDescriptors.size())
            return nullptr;
        return &supportedPrimitiveDescriptors[selectedPrimitiveDescriptorIndex];
    }

    inline MKLDNNPrimitiveDescInfo* getSelectedPrimitiveDescriptor() {
        if (selectedPrimitiveDescriptorIndex < 0 ||
            selectedPrimitiveDescriptorIndex >= supportedPrimitiveDescriptors.size())
            return nullptr;
        return &supportedPrimitiveDescriptors[selectedPrimitiveDescriptorIndex];
    }

    void selectPrimitiveDescriptorByIndex(int index) {
        if (index < 0 || index >= supportedPrimitiveDescriptors.size())
            selectedPrimitiveDescriptorIndex = -1;
        else
            selectedPrimitiveDescriptorIndex = index;
    }

    std::string getPrimitiveDescriptorType();

    PerfCount &PerfCounter() { return perfCounter; }

    const mkldnn::memory::data_type& getInputDataType() const {
        return inputDataType;
    }
    const mkldnn::memory::data_type& getOutputDataType() const {
        return outputDataType;
    }

    void setDynamicBatchLim(int lim) {
        dynBatchLim = lim;
    }

    virtual void resolveNotAllocatedEdges();
    virtual void execute(mkldnn::stream strm);
    virtual void initSupportedPrimitiveDescriptors(const mkldnn::engine &engine);
    virtual void createPrimitive() = 0;

    virtual void selectOptimalPrimitiveDescriptor();

    virtual void createDescriptor(mkldnn::memory::data_type inputDataType, mkldnn::memory::data_type outputDataType) = 0;
    virtual bool created() = 0;
    virtual bool created(const MKLDNNExtensionManager::Ptr& extMgr) {
        return created();
    }
    virtual void initEdges();

    template <class PD, class D, typename FPD = bool>
    PD createPrimitiveDescriptor(const mkldnn::primitive_attr &attr = mkldnn::primitive_attr()) {
        auto descsEqual = [](const std::vector<MKLDNNMemoryDesc>& srcDescs,
                               const std::vector<MKLDNNMemoryDesc>& selectedDescs) {
            if (srcDescs.empty() && selectedDescs.empty())
                return true;
            if (srcDescs.empty() || selectedDescs.empty())
                return false;
            for (size_t i = 0; i < srcDescs.size() && i < selectedDescs.size(); i++) {
                if (srcDescs[i] != selectedDescs[i] && srcDescs[i])
                    return false;
            }
            return true;
        };

        const MKLDNNPrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
        if (selected_pd == nullptr)
            THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set for node " << getName() << ".";
        prepareMemory(selected_pd);

        for (auto& desc : descs) {
            try {
                mkldnn::primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(selected_pd->getEngine(),
                                                                                              attr);
                do {
                    std::vector<MKLDNNMemoryDesc> srcDescs;
                    for (size_t i = 0; i < desc.inputNumbers() && srcMemDesc; i++)
                        srcDescs.push_back(srcMemDesc(itpd, i));

                    std::vector<MKLDNNMemoryDesc> intDescs;
                    for (auto &it : internalBlobDesc)
                        intDescs.push_back(it(itpd, 0));

                    std::vector<MKLDNNMemoryDesc> dstDescs;
                    for (size_t i = 0; i < desc.outputNumbers() && dstMemDesc; i++)
                        dstDescs.push_back(dstMemDesc(itpd, i));

                    impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str().c_str());

                    if (impl_type == selected_pd->getImplementationType() &&
                        descsEqual(srcDescs, selected_pd->getInputDescs()) &&
                        descsEqual(dstDescs, selected_pd->getOutputDescs()) &&
                        descsEqual(intDescs, selected_pd->getInternalDescs())) {
                        PD prim_desc = createPd<PD, D, FPD>(desc, selected_pd->getEngine());
                        itpd.getPrimitiveDescriptor(prim_desc);
                        return prim_desc;
                    }
                } while (itpd.next());
            } catch (std::exception e) {
                // it throw exception in case of no implementation found
                continue;
            }
        }

        THROW_IE_EXCEPTION << "Primitive descriptor was not found for node " << getName() << ".";
    }

private:
    std::vector<MKLDNNEdgeWeakPtr> parentEdges;
    std::vector<MKLDNNEdgeWeakPtr> childEdges;

    InferenceEngine::CNNLayerPtr cnnLayer;

    std::string name;
    const std::string typeStr;
    Type type;

    std::string typeToStr(Type type);

    PerfCount perfCounter;

    // TODO: It is necessary only in order to avoid modifications of cnnLayers and original topology
    std::vector<MKLDNNDims> outDims;
    std::vector<MKLDNNDims> inDims;

    int selectedPrimitiveDescriptorIndex = -1;

    bool isEdgesEmpty(const std::vector<MKLDNNEdgeWeakPtr>& edges) const;

    class Registry {
    public:
        typedef std::function<MKLDNNNode *(const Type type, const std::string &name)> CreatorByNameFunction;
        typedef std::function<MKLDNNNode *(const InferenceEngine::CNNLayerPtr& layer)> CreatorByLayerFunction;

        static MKLDNNNode *CreateNode(const Type type, const std::string &name, const MKLDNNExtensionManager::Ptr& extMgr);
        static MKLDNNNode *CreateNode(const InferenceEngine::CNNLayerPtr& layer, const MKLDNNExtensionManager::Ptr& extMgr);

        static void RegisterNode(CreatorByNameFunction f);
        static void RegisterNode(CreatorByLayerFunction f);
    private:
        static std::vector<CreatorByNameFunction> _dataByName;
        static std::vector<CreatorByLayerFunction> _dataByLayer;
    };

    template <class PD, class D, typename FPD>
    typename std::enable_if<!std::is_same<FPD, bool>::value, PD>::type
    createPd(MKLDNNDescriptor desc, const mkldnn::engine& engine) {
        std::shared_ptr<D> selected_desc_ptr = desc;
        std::shared_ptr<FPD> backward_prim_desc_ptr = desc;
        return PD(*selected_desc_ptr, engine, *backward_prim_desc_ptr);
    }

    template <class PD, class D, typename FPD>
    typename std::enable_if<std::is_same<FPD, bool>::value, PD>::type
    createPd(MKLDNNDescriptor desc, const mkldnn::engine& engine) {
        std::shared_ptr<D> selected_desc_ptr = desc;
        return PD(*selected_desc_ptr, engine);
    }

    void prepareMemory(const MKLDNNPrimitiveDescInfo *selected_pd);


protected:
    void setType(Type type) {
        this->type = type;
    }

    typedef std::function<MKLDNNMemoryDesc (mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx)>
            GetPrimitiveMemoryFormatFunc;

    GetPrimitiveMemoryFormatFunc srcMemDesc;
    GetPrimitiveMemoryFormatFunc dstMemDesc;
    std::vector<GetPrimitiveMemoryFormatFunc> internalBlobDesc;

    std::vector <MKLDNNNodePtr> fusedWith;
    std::vector <MKLDNNNodePtr> mergedWith;

    MKLDNNNode(Type type, const std::string &name);
    explicit MKLDNNNode(InferenceEngine::CNNLayerPtr layer);

    bool permanent = false;
    bool temporary = false;
    mkldnn::memory::data_type inputDataType;
    mkldnn::memory::data_type outputDataType;
    int dynBatchLim = 0;
    bool constant;
    std::vector<InferenceEngine::Blob::Ptr> internalBlobs;
    std::vector<MKLDNNMemoryPtr> internalBlobMemory;
    std::vector<MKLDNNPrimitiveDescInfo> supportedPrimitiveDescriptors;
    std::shared_ptr<mkldnn::primitive> prim;
    std::vector<MKLDNNDescriptor> descs;

    friend class MKLDNNEdge;
    friend class MKLDNNGraph;
    friend class MKLDNNGraphOptimizer;

    virtual void selectPreferPrimitiveDescriptor(const std::vector<impl_desc_type>& priority);
    virtual bool initAsInPlace();

    static const std::vector<impl_desc_type> primitivesPriority;

    std::string primitiveDescriptorTypeToString(impl_desc_type type);

    std::vector<mkldnn::memory::format> getAvailableFormatsForDims(const MKLDNNDims& dims) const;
    int batchToProcess(int b) { return dynBatchLim == 0 ? b : std::min<int>(b, dynBatchLim); }

    InferenceEngine::Blob::Ptr createInternalBlob(InferenceEngine::SizeVector dims, bool weights);

    template<typename To>
    class Register {
    public:
        Register() {
            Registry::RegisterNode(
                Registry::CreatorByNameFunction([](const Type type, const std::string &name) -> MKLDNNNode * {
                    return new To(type, name); } ) );
            Registry::RegisterNode(
                Registry::CreatorByLayerFunction([](const InferenceEngine::CNNLayerPtr& layer) -> MKLDNNNode * {
                    return new To(layer); } ) );
        }
    };
};

template <typename T, typename U>
inline T div_up(const T a, const U b) {
    assert(b);
    return (a + b - 1) / b;
}

template <typename T, typename U>
inline T rnd_up(const T a, const U b) {
    return div_up(a, b) * b;
}

MKLDNNDims autoBlockingDims(const MKLDNNDims &dims, mkldnn::memory::format fmt);
}  // namespace MKLDNNPlugin
