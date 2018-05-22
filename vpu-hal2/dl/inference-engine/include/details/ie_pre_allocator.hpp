// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <details/ie_exception.hpp>
#include "ie_allocator.hpp"
#include <memory>

namespace InferenceEngine {
namespace details {
/*
 * @class PreAllocator
 * @brief This is a helper class to wrap external memory
 */
class PreAllocator : public IAllocator {
    void * _actualData;
    size_t _sizeInBytes;

 public:
    PreAllocator(void *ptr, size_t bytes_size)
        : _actualData(ptr), _sizeInBytes(bytes_size) {}
    /**
     * @brief Locks a handle to heap memory accessible by any memory manipulation routines
     * @return The generic pointer to a memory buffer
     */
    void * lock(void * handle, LockOp = LOCK_FOR_WRITE)  noexcept override {
        if (handle != _actualData) {
            return nullptr;
        }
        return handle;
    }
    /**
     * @brief The PreAllocator class does not utilize this function
     * @param handle Memory handle to unlock
     */
    void  unlock(void * handle) noexcept override {}

    /**
     * @brief Returns a pointer to preallocated memory
     * @param size Size in bytes
     * @return A handle to the preallocated memory or nullptr
     */
    void * alloc(size_t size) noexcept override {
        if (size <= _sizeInBytes) {
            return _actualData;
        }

        return this;
    }
    /**
     * @brief The PreAllocator class cannot release the handle
     * @param handle Memory handle to release
     * @return false
     */
    bool   free(void* handle) noexcept override { return false;}

    /**
     * @brief Deletes current allocator. 
     * Can be used if a shared_from_irelease pointer is used
     */
    void Release() noexcept override {
        delete this;
    }

 protected:
    virtual ~PreAllocator() = default;
};

/**
 * @brief Creates a special allocator that only works on external memory
 * @param ptr Pointer to preallocated memory
 * @param size Number of elements allocated
 * @return A new allocator
 */

template <class T>
std::shared_ptr<IAllocator>  make_pre_allocator(T *ptr, size_t size) {
    return shared_from_irelease(new PreAllocator(ptr, size * sizeof(T)));
}

}  // namespace details
}  // namespace InferenceEngine