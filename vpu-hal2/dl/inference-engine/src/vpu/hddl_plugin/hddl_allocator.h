#pragma once

#include <map>
#include <memory>

#include "ie_allocator.hpp"
#include "hddl_api.h"
#include "hddl_common.h"
#include "vpu_logger.h"

namespace VPU {
namespace HDDLPlugin {

class HDDLAllocator : public InferenceEngine::IAllocator {
public:
    HDDLAllocator(const Common::LoggerPtr& log);

    // FIXME: return some status
    void reserveHddlBuffer(size_t size);

    void *lock(void *handle, InferenceEngine::LockOp = InferenceEngine::LOCK_FOR_WRITE) noexcept override;
    void unlock(void *handle) noexcept override;

    void *alloc(size_t size) noexcept override;
    bool free(void* handle) noexcept override;

    HddlBuffer *getHddlBufferByPointer(void *ptr);
    bool isBlobPlacedWholeBuffer(void *ptr);
    bool isIonBuffer(void *ptr);
    HddlAllocatorHandle getAllocatorHandle();

    void Release() noexcept;

private:
    struct MemoryPoolItem {
        HddlBuffer buffer;
        size_t refCount;
    };

    std::map<void *, MemoryPoolItem> _memoryPool;
    MemoryPoolItem *lastReservedItem = nullptr;
    size_t offsetToAllocateMemory = 0;

    HddlAllocatorHandle _allocatorHandle;
    Common::LoggerPtr _log;
};

using HDDLAllocatorPtr = std::shared_ptr<HDDLAllocator>;

}  // namespace HDDLPlugin
}  // namespace VPU
