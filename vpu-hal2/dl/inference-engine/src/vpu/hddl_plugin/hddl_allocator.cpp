#include "hddl_allocator.h"


using namespace VPU::HDDLPlugin;

HDDLAllocator::HDDLAllocator(const Common::LoggerPtr& log): _log(log) {
    int status = hddlAllocatorInit(& _allocatorHandle);
    if (status != HDDL_ERROR_NONE) {
        LOG_WARNING("hddlAllocatorInit failed");
    }
}

void HDDLAllocator::Release() noexcept {
    for (auto &item : _memoryPool) {
        HddlBuffer *buffer = &item.second.buffer;
        int status = hddlAllocatorFree(_allocatorHandle, buffer);
        if (status != HDDL_ERROR_NONE) {
            LOG_WARNING("Error: hddlAllocatorGetBufferData failed");
        }
    }
    _memoryPool.clear();

    int status = hddlAllocatorDeinit(_allocatorHandle);
    if (status != HDDL_ERROR_NONE) {
        LOG_WARNING("hddlAllocatorInit failed");
    }
}

void *HDDLAllocator::lock(void *handle, InferenceEngine::LockOp lockOp) noexcept {
    (void)lockOp;
    return handle;
}

void HDDLAllocator::unlock(void *handle) noexcept {
    (void)handle;
}

void HDDLAllocator::reserveHddlBuffer(size_t size) {
    MemoryPoolItem item;
    item.refCount = 0;
    int status = hddlAllocatorAlloc(_allocatorHandle, size, &item.buffer);
    if (status != HDDL_ERROR_NONE) {
        LOG_WARNING("hddlAllocatorAlloc failed");
    }

    void *data = nullptr;
    status = hddlAllocatorGetBufferData(_allocatorHandle, &item.buffer, &data);
    if (status != HDDL_ERROR_NONE) {
        LOG_WARNING("hddlAllocatorGetBufferData failed");
    }

    if (data) {
        _memoryPool[data] = item;
        lastReservedItem = &_memoryPool[data];
        offsetToAllocateMemory = 0;
    }
}

void *HDDLAllocator::alloc(size_t size) noexcept {
    size_t reservedBufferSize = 0;
    int status = hddlAllocatorGetBufferSize(_allocatorHandle, &lastReservedItem->buffer, &reservedBufferSize);
    if (status != HDDL_ERROR_NONE) {
        LOG_WARNING("hddlAllocatorGetBufferSize failed");
    }

    if (offsetToAllocateMemory + size <= reservedBufferSize) {
        uint8_t *data = nullptr;
        hddlAllocatorGetBufferData(_allocatorHandle, &lastReservedItem->buffer, reinterpret_cast<void **>(&data));
        uint8_t *allocatingMemory = data + offsetToAllocateMemory;
        offsetToAllocateMemory += size;
        lastReservedItem->refCount++;
        return reinterpret_cast<void *>(allocatingMemory);
    } else {
        LOG_WARNING("Could not allocate memory in HddlBuffer");
    }

    return nullptr;
}

bool HDDLAllocator::free(void *handle) noexcept {
    auto it = _memoryPool.find(handle);
    if (it != _memoryPool.end()) {
        auto item = it->second;
        item.refCount--;
        if (item.refCount == 0) {
            int status = hddlAllocatorFree(_allocatorHandle, &item.buffer);
            if (status != HDDL_ERROR_NONE) {
                LOG_WARNING("Error: hddlAllocatorGetBufferData failed");
            }

            _memoryPool.erase(it);
        }
        return true;
    }

    return false;
}

HddlBuffer *HDDLAllocator::getHddlBufferByPointer(void *ptr) {
    auto it = _memoryPool.find(ptr);

    return it != _memoryPool.end() ? &it->second.buffer : nullptr;
}

bool HDDLAllocator::isBlobPlacedWholeBuffer(void *ptr) {
    auto it = _memoryPool.find(ptr);

    if (it != _memoryPool.end()) {
        return it->second.refCount == 1 ? true : false;
    }

    return false;
}

bool HDDLAllocator::isIonBuffer(void *ptr) {
    return getHddlBufferByPointer(ptr) != nullptr;
}

HddlAllocatorHandle HDDLAllocator::getAllocatorHandle() {
    return _allocatorHandle;
}
