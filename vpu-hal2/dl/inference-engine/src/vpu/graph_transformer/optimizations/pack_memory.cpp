//
// INTEL CONFIDENTIAL
// Copyright 2018 Intel Corporation.
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

#include "graph_transformer_impl.hpp"
#include <unordered_map>
#include <list>
#include <unordered_set>
#include <algorithm>
#include "vpu_logger.h"


#ifdef NNLOG
#include <android/log.h>
#include <cutils/log.h>
#endif

namespace {

uint32_t calcAbsParentOffset(const VpuDims& offset, const VpuStrides& strides) {
    return   offset[Dim::X] * strides[Dim::X]
           + offset[Dim::Y] * strides[Dim::Y]
           + offset[Dim::Z] * strides[Dim::Z];
}

uint32_t calcDataTotalSize(const VpuDataHandle& data) {
    if (data->order == orderYXZ) {
        return data->strides[Dim::Y] * data->dims[Dim::Y];
    } else if (data->order == orderZYX) {
        return data->strides[Dim::Z] * data->dims[Dim::Z];
    } else {
        THROW_IE_EXCEPTION << "[VPU] Unsupported data order " << mvTensorStorageOrderToStr(data->order);
    }
}

bool isStageCMXFree(const VpuStageHandle& stage) {
    if (isHwStage(stage))
        return true;

    if (stage->type == kCopyMakeBorderCHW) {
        auto input = stage->inputs[0];
        auto output = stage->outputs[0];
        return input->dims[Dim::X] == output->dims[Dim::X] &&
               input->dims[Dim::Y] == output->dims[Dim::Y];
    }

    if (stage->type == kCopy) {
        return true;
    }

    return false;
}

class VpuAllocator {
public:
    struct Chunk {
        IndexCodes index;
        uint32_t offset;
        uint32_t padding;
        uint32_t size;
        uint32_t inuse;
        VpuAllocator* allocator;
    };

    struct FreeMemory {
        uint32_t offset;
        uint32_t size;
    };

    VpuAllocator(bool memOptimization, IndexCodes index, uint32_t maxSize)
        : _memOptimization(memOptimization), _index(index), _maxSize(maxSize) {
    }

    Chunk* find(VpuDataHandle data) {
        auto memMapIt = _memMap.find(data);

        if (memMapIt != _memMap.end()) {
            return &memMapIt->second;
        }

        return nullptr;
    }

    Chunk* allocate(uint32_t size, uint32_t padding, uint32_t inuse,
                    VpuDataHandle data) {
        Chunk* chunk = nullptr;

#ifdef NNLOG
    ALOGI("[VPU] GraphTransformer : _memOptimization = %d size = %u", _memOptimization,
               static_cast<uint32_t>(size));
#endif

        if (_memOptimization) {

            auto minMemIt = _memPool.end();

            for (auto memPoolIt = _memPool.begin(); memPoolIt != _memPool.end(); ++memPoolIt) {
                if (memPoolIt->size >= size) {
                    minMemIt = memPoolIt;
                    break;
                }
            }

            if (minMemIt != _memPool.end()) {
                auto res = _memMap.insert({data, {_index, minMemIt->offset + minMemIt->size - size, padding, size, inuse, this}});
                assert(res.second);

                chunk = &res.first->second;

                minMemIt->size -= size;
                if (minMemIt->size == 0) {
                    _memPool.erase(minMemIt);
                }
            }
        }

        if (chunk == nullptr) {
            if (_memOffset + size <= _maxSize) {
                auto res = _memMap.insert({data, {_index, _memOffset, padding, size, inuse, this}});
                assert(res.second);

                chunk = &res.first->second;

                _memOffset += size;
            }
        }

        if (chunk) {
            _memUsed = std::max(_memUsed, chunk->offset + chunk->size);
        }

        return chunk;
    }

    void free(Chunk* chunk) {
        if (chunk == nullptr)
            return;

        assert(chunk->allocator == this);
        assert(chunk->inuse > 0);

        if (--chunk->inuse == 0) {
            FreeMemory newMem{chunk->offset, chunk->size};
            while (true) {
                bool found = false;

                for (auto memPoolIt = _memPool.begin(); memPoolIt != _memPool.end(); ++memPoolIt) {
                    if (newMem.offset + newMem.size == memPoolIt->offset) {
                        // [newMem][*memPoolIt] case
                        // extend newMem to and remove memPoolIt
                        newMem.size += memPoolIt->size;
                        _memPool.erase(memPoolIt);
                        found = true;
                        break;
                    } else if (memPoolIt->offset + memPoolIt->size == newMem.offset) {
                        // [*memPoolIt][newMem] case
                        // extend newMem to and remove memPoolIt
                        newMem.offset = memPoolIt->offset;
                        newMem.size += memPoolIt->size;
                        _memPool.erase(memPoolIt);
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    if (newMem.offset + newMem.size == _memOffset) {
                        _memOffset = newMem.offset;
                    } else {
                        _memPool.push_back(newMem);
                    }

                    break;
                }
            }
        }
    }

    void check() {
        if (!_memPool.empty() || _memOffset > 0) {
            THROW_IE_EXCEPTION << "[VPU] Blob memory packing failed";
        }
    }

    uint32_t memUsed() const {
        return _memUsed;
    }

private:
    bool _memOptimization;
    IndexCodes _index;
    uint32_t _maxSize;

    uint32_t _memOffset = 0;
    uint32_t _memUsed = 0;

    std::unordered_map<VpuDataHandle, Chunk, VpuDataHandleHash> _memMap;
    std::list<FreeMemory> _memPool;
};

const uint32_t MIN_HW_PADDING = 0u;

uint32_t getStageRequiredOutputPadding(VpuStageHandle stage, VpuDataHandle parent) {
    bool isFirstOutput = stage->outputs[0] == parent;
    if (!isFirstOutput) {
        loopOverSubData(parent, [&isFirstOutput, stage](VpuDataHandle subData) {
            if (!isFirstOutput)
                isFirstOutput = stage->outputs[0] == subData;
        });
    }

    if (!isFirstOutput)
        return 0;

    uint32_t requiredPadding = 0;

    if (stage->type == kMyriadXHwConvolution) {
        auto hwStage = stage.dynamicCast<VpuMyriadXHwConvolutionStage>();
        assert(hwStage != nullptr);

        assert(stage->outputs.size() >= 1);
        auto output = stage->outputs[0];
        assert(output != nullptr);

        auto origDimZ = output->dims[Dim::Z];
        auto newDimZ = hwStage->newOutputDimZ;
        assert(newDimZ >= origDimZ);

        auto diff = newDimZ - origDimZ;
        diff = std::max(diff, MIN_HW_PADDING);

        requiredPadding = diff * output->strides[Dim::Z];
    } else if (stage->type == kMyriadXHwFCL) {
        auto hwStage = stage.dynamicCast<VpuMyriadXHwFullyConnectedStage>();
        assert(hwStage != nullptr);

        assert(stage->outputs.size() >= 1);
        auto output = stage->outputs[0];
        assert(output != nullptr);

        auto origDimZ = output->dims[Dim::Z];
        auto newDimZ = hwStage->newOutputDimZ;
        assert(newDimZ >= origDimZ);

        auto diff = newDimZ - origDimZ;
        diff = std::max(diff, MIN_HW_PADDING);

        requiredPadding = diff * output->strides[Dim::Z];
    } else if (stage->type == kMyriadXHwPooling) {
        auto hwStage = stage.dynamicCast<VpuMyriadXHwPoolingStage>();
        assert(hwStage != nullptr);

        assert(stage->outputs.size() >= 1);
        auto output = stage->outputs[0];
        assert(output != nullptr);

        auto origDimZ = output->dims[Dim::Z];
        auto newDimZ = hwStage->newOutputDimZ;
        assert(newDimZ >= origDimZ);

        auto diff = newDimZ - origDimZ;
        diff = std::max(diff, MIN_HW_PADDING);

        requiredPadding = diff * output->strides[Dim::Z];
    } else if (stage->type == kMaxPool || stage->type == kAvgPool) {
        // For Max/Avg pooling out of buffer writing workaround
        assert(stage->inputs.size()  >= 1);
        assert(stage->outputs.size() >= 1);
        auto input  = stage->inputs[0];
        auto output = stage->outputs[0];
        assert(output != nullptr);
        assert(input != nullptr);

        int origDimY = static_cast<int>(input->dims[Dim::Y]);
        int newDimY = static_cast<int>(output->dims[Dim::Y]);
        int StY = static_cast<int>(output->strides[Dim::Y]);

        requiredPadding = ((origDimY - newDimY) > 0)
            ? (origDimY - newDimY) * StY : 0;
    }

    return requiredPadding;
}

uint32_t getStageRequiredInputPadding(VpuStageHandle stage, VpuDataHandle parent) {
    bool isFirstInput = stage->inputs[0] == parent;
    if (!isFirstInput) {
        loopOverSubData(parent, [&isFirstInput, stage](VpuDataHandle subData) {
            if (!isFirstInput)
                isFirstInput = stage->inputs[0] == subData;
        });
    }

    if (!isFirstInput)
        return 0;

    uint32_t requiredPadding = 0;

    if (stage->type == kMyriadXHwConvolution) {
        auto hwStage = stage.dynamicCast<VpuMyriadXHwConvolutionStage>();
        assert(hwStage != nullptr);

        auto input = stage->inputs[0];
        assert(input != nullptr);

        auto origDimZ = input->dims[Dim::Z];
        auto newDimZ = hwStage->newInputDimZ;
        assert(newDimZ >= origDimZ);

        auto diff = newDimZ - origDimZ;
        diff = std::max(diff, MIN_HW_PADDING);

        requiredPadding = diff * input->strides[Dim::Z];
    } else if (stage->type == kMyriadXHwFCL) {
        auto hwStage = stage.dynamicCast<VpuMyriadXHwFullyConnectedStage>();
        assert(hwStage != nullptr);

        auto input = stage->inputs[0];
        assert(input != nullptr);

        auto origDimZ = input->dims[Dim::Z];
        auto newDimZ = hwStage->newInputDimZ;
        assert(newDimZ >= origDimZ);

        auto diff = newDimZ - origDimZ;
        diff = std::max(diff, MIN_HW_PADDING);

        requiredPadding = diff * input->strides[Dim::Z];
    } else if (stage->type == kMyriadXHwPooling) {
        auto hwStage = stage.dynamicCast<VpuMyriadXHwPoolingStage>();
        assert(hwStage != nullptr);

        auto input = stage->inputs[0];
        assert(input != nullptr);

        auto origDimZ = input->dims[Dim::Z];
        auto newDimZ = hwStage->newOutputDimZ;
        assert(newDimZ >= origDimZ);

        auto diff = newDimZ - origDimZ;
        diff = std::max(diff, MIN_HW_PADDING);

        requiredPadding = diff * input->strides[Dim::Z];
    }

    return requiredPadding;
}

}  // namespace

void GraphTransformerImpl::packMemory() {
    std::unordered_set<VpuDataHandle, VpuDataHandleHash> processedData;

    VpuAllocator cmxAllocator(_blobConfig.memoryOptimization, IndexCMX, _blobConfig.cmxBufferSize);
    VpuAllocator ddrAllocator(_blobConfig.memoryOptimization, IndexBSS, 512u * 1024u * 1024u);

#ifdef NNLOG
    ALOGI("[VPU] GraphTransformer packMemory _blobConfig.memoryOptimization = %d",_blobConfig.memoryOptimization);
#endif
    LOG_INFO("[VPU] GraphTransformer packMemory _blobConfig.memoryOptimization = %d",_blobConfig.memoryOptimization);

    auto findChunk = [&cmxAllocator, &ddrAllocator](VpuDataHandle data) {
        if (auto chunk = cmxAllocator.find(data))
            return chunk;

        return ddrAllocator.find(data);
    };

    auto allocateChunk = [&cmxAllocator, &ddrAllocator](
            bool canUseCMX,
            uint32_t size, uint32_t padding, uint32_t inuse,
            VpuDataHandle data) {
        if (canUseCMX) {
            if (auto chunk = cmxAllocator.allocate(size, padding, inuse, data))
                return chunk;
        }

        return ddrAllocator.allocate(size, padding, inuse, data);
    };

    // Allocate space for BSS/CMX data

    for (auto stageIt = _stages.begin(); stageIt != _stages.end(); ++stageIt) {
        auto stage = *stageIt;
        assert(stage != nullptr);

        #ifdef NNLOG
            ALOGI("[VPU] GraphTransformer packMemory stage->optimized = %d",stage->optimized);
        #endif
            LOG_INFO("[VPU] GraphTransformer packMemory stage->optimized = %d",stage->optimized);

        if (stage->optimized)
            continue;

        // allocate outputs

        for (const auto& output : stage->outputs) {
            assert(output != nullptr);

            if (output->index != IndexBSS && output->index != IndexCMX)
                continue;

            if (processedData.find(output) != processedData.end())
                continue;

            auto parent = getDataTopParent(output);

            auto chunk = findChunk(parent);

            if (chunk != nullptr) {
                if (parent == output) {
                    THROW_IE_EXCEPTION << "[VPU] Trying to allocate the same data " << output->name << " twice";
                }

                loopOverSubData(parent, [parent, &processedData](VpuDataHandle subData) {
                    if (processedData.find(subData) != processedData.end())
                        return;

                    if (subData->parent == nullptr) {
                        THROW_IE_EXCEPTION << "[VPU] in function " << __PRETTY_FUNCTION__ << ": parent of VPU data handle not defined.";
                    }

                    subData->index = parent->index;
                    subData->offset =   subData->parent->offset
                                      + calcAbsParentOffset(subData->offsetFromParent, subData->strides);

                    processedData.insert(subData);
                });
            } else {
                // Get list of all consumers and producers

                std::unordered_set<VpuStageHandle, VpuStageHandleHash> consumers;
                for (const auto& consumer : parent->consumers) {
                    assert(consumer != nullptr);
                    if (!consumer->optimized)
                        consumers.insert(consumer);
                }
                loopOverSubData(parent, [&consumers](VpuDataHandle subData) {
                    for (const auto& consumer : subData->consumers) {
                        assert(consumer != nullptr);
                        if (!consumer->optimized)
                            consumers.insert(consumer);
                    }
                });

                if (consumers.empty()) {
                    THROW_IE_EXCEPTION << "[VPU] Stage " << stage->name
                                       << " have output which is not used " << output->name
                                       << " by any other stage";
                }

                std::unordered_set<VpuStageHandle, VpuStageHandleHash> producers;
                producers.insert(stage);
                if (stage->parentOp != nullptr && !stage->parentOp->optimized)
                    producers.insert(stage->parentOp);
                if (stage->postOp != nullptr && !stage->postOp->optimized)
                    producers.insert(stage->postOp);
                loopOverSubData(parent, [&producers](VpuDataHandle subData) {
                    if (subData->producer != nullptr && !subData->producer->optimized) {
                        producers.insert(subData->producer);
                    }
                });

                // Check if we can use CMX

                bool canUseCMX = _blobConfig.useCmxBuffers && isStageCMXFree(stage);
                for (const auto& consumer : consumers) {
                    if (!canUseCMX)
                        break;

                    if (!isStageCMXFree(consumer)) {
                        canUseCMX = false;
                        break;
                    } else {
                        // Check that between current stage and consumer there is no stages that use CMX

                        auto nextIt = stageIt;
                        ++nextIt;

                        auto consumerIt = std::find_if(nextIt, _stages.end(),
                                                       [consumer](const VpuStagePtr &ptr) {
                                                           return ptr.get() == consumer.get();
                                                       });
                        if (consumerIt == _stages.end()) {
                            THROW_IE_EXCEPTION << "[VPU] Internal error (invalid list of stages)";
                        }

                        for (auto it = nextIt; it != consumerIt; ++it) {
                            auto middleStage = *it;
                            if (!middleStage->optimized && !isStageCMXFree(middleStage)) {
                                canUseCMX = false;
                                break;
                            }
                        }
                    }
                }
                for (const auto& producer : producers) {
                    if (!canUseCMX)
                        break;

                    if (!isStageCMXFree(producer)) {
                        canUseCMX = false;
                        break;
                    }
                }

                // Get required padding size

                uint32_t paddingSize = 0;
                for (const auto& producer : producers) {
                    auto curPadding = getStageRequiredOutputPadding(producer, parent);
                    paddingSize = std::max(paddingSize, curPadding);
                }
                for (const auto& consumer : consumers) {
                    auto curPadding = getStageRequiredInputPadding(consumer, parent);
                    paddingSize = std::max(paddingSize, curPadding);
                }

                // Align padding to 16 bytes
                paddingSize = alignVal(paddingSize, 16u);

                // Calculate final buffer size

                auto dataSize = calcDataTotalSize(parent);

                auto bufferSize = alignVal(dataSize + 2 * paddingSize, DATA_ALIGNMENT);
                LOG_DEBUG("[VPU] GraphTransformer : data %s dataSize=%u paddingSize=%u bufferSize=%u",
                          parent->name.c_str(),
                          static_cast<uint32_t>(dataSize),
                          static_cast<uint32_t>(paddingSize),
                          static_cast<uint32_t>(bufferSize));

                LOG_INFO("[VPU] GraphTransformer DATA_ALIGNMENT : data %s dataSize=%u paddingSize=%u bufferSize=%u",
                          parent->name.c_str(),
                          static_cast<uint32_t>(dataSize),
                          static_cast<uint32_t>(paddingSize),
                          static_cast<uint32_t>(bufferSize));

#ifdef NNLOG
                ALOGI("[VPU] GraphTransformer DATA_ALIGNMENT : data %s dataSize=%u paddingSize=%u bufferSize=%u",
                          parent->name.c_str(),
                          static_cast<uint32_t>(dataSize),
                          static_cast<uint32_t>(paddingSize),
                          static_cast<uint32_t>(bufferSize));
#endif
                // TODO: investigate this; makes hw googlenet stable
                if (bufferSize > CMX_BUFFER_SIZE_LIMIT)
                    canUseCMX = false;

                // Allocate chunk

                chunk = allocateChunk(canUseCMX, bufferSize, paddingSize, consumers.size(), parent);
                if (chunk == nullptr) {
                    THROW_IE_EXCEPTION << "[VPU] Could not allocate memory buffer for " << parent->name;
                }

                parent->index = chunk->index;
                parent->offset = chunk->offset + chunk->padding;
                if (parent->index == IndexCMX) {
                    parent->offset += _blobConfig.cmxBufferStart;
                }
                loopOverSubData(parent, [parent, &processedData](VpuDataHandle subData) {
                    if (processedData.find(subData) != processedData.end())
                        return;

                    if (subData->parent == nullptr) {
                        THROW_IE_EXCEPTION << "[VPU] in function " << __PRETTY_FUNCTION__ << ": parent of VPU data handle not defined.";
                    }

                    subData->index = parent->index;
                    subData->offset =   subData->parent->offset
                                      + calcAbsParentOffset(subData->offsetFromParent, subData->strides);

                    processedData.insert(subData);
                });
            }

            processedData.insert(parent);
            processedData.insert(output);
        }

        // check inputs

        for (const auto& input : stage->inputs) {
            assert(input != nullptr);

            if (input->index != IndexBSS && input->index != IndexCMX)
                continue;

            auto parent = getDataTopParent(input);

            auto chunk = findChunk(parent);

            if (chunk == nullptr) {
                auto producer = parent->producer;
                if (producer == nullptr || !producer->optimized) {
                    THROW_IE_EXCEPTION << "[VPU] Could not allocate memory buffer for " << input->name;
                }

                continue;
            }

            if (chunk->inuse == 0) {
                THROW_IE_EXCEPTION << "[VPU] Input " << input->name << " is not used";
            }

            assert(chunk->allocator != nullptr);
            chunk->allocator->free(chunk);
        }
    }

    // Self-check

    cmxAllocator.check();
    ddrAllocator.check();

    // Pack Blob data

    _blobTotalDataSize = 0;
    for (auto& data : _datas) {
        if (data->index == IndexBlob && data->writer != nullptr) {
            auto dataSize = data->writer->byteSize();
            auto alignedSize = alignVal(dataSize, static_cast<size_t>(WEIGHTS_ALIGNMENT));
            data->offset = _blobTotalDataSize;
            _blobTotalDataSize += alignedSize;
        }
    }
    LOG_INFO("[VPU] GraphTransformer WEIGHTS_ALIGNMENT: _blobTotalDataSize = %u",
             static_cast<uint32_t>(_blobTotalDataSize));

#ifdef NNLOG
    ALOGI("[VPU] GraphTransformer WEIGHTS_ALIGNMENT : _blobTotalDataSize = %u",
             static_cast<uint32_t>(_blobTotalDataSize));
#endif

    // Calculate offset for subData

    for (const auto& data : _datas) {
        loopOverSubData(data, [&processedData](VpuDataHandle subData) {
            if (processedData.find(subData) != processedData.end())
                return;

            if (subData->parent == nullptr) {
                THROW_IE_EXCEPTION << "[VPU] in function " << __PRETTY_FUNCTION__ << ": parent of VPU data handle not defined.";
            }

            subData->index = subData->parent->index;
            subData->offset =   subData->parent->offset
                              + calcAbsParentOffset(subData->offsetFromParent, subData->strides);

            processedData.insert(subData);
        });
    }

    // Allocate temporay buffers

    uint32_t maxTempBufSize = 0u;

    for (const auto& stage : _stages) {
        assert(stage != nullptr);

        if (stage->optimized)
            continue;

        if (stage->buffer != nullptr) {
            stage->buffer->offset = ddrAllocator.memUsed();

            maxTempBufSize = std::max(maxTempBufSize, calcDataTotalSize(stage->buffer));
        }
    }

    _bssMemSize = ddrAllocator.memUsed() + maxTempBufSize;

    LOG_INFO("[VPU] GraphTransformer : DDR memory usage = %u CMX memory usage = %u",
             static_cast<uint32_t>(_bssMemSize),
             static_cast<uint32_t>(cmxAllocator.memUsed()));
#ifdef NNLOG
    ALOGI("[VPU] GraphTransformer : DDR memory usage = %u CMX memory usage = %u",
             static_cast<uint32_t>(_bssMemSize),
             static_cast<uint32_t>(cmxAllocator.memUsed()));
#endif
}
