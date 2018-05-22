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

#include <details/ie_irelease.hpp>
#include <ie_api.h>

namespace InferenceEngine {

/**
 * @brief Allocator handle mapping type
 */
enum LockOp {
    LOCK_FOR_READ = 0,
    LOCK_FOR_WRITE
};

/**
 * @class IAllocator
 * @brief Allocator concept to be used for memory management, used as part of TBlob
 */
class IAllocator  : public details::IRelease {
public:
    /**
     * @brief Mapping handle to heap memory accessible by any memory manipulation routines
     * @return Generic pointer to memory
     */
    virtual void * lock(void * handle, LockOp = LOCK_FOR_WRITE)  noexcept = 0;
    /**
     * @brief Unmaps memory by handle, multiple sequential mappings of same handle suppose to get same result
     * while no ref counter supported
     */
    virtual void  unlock(void * handle) noexcept = 0;
    /**
     * @brief Allocates memory
     * @param size Size in bytes to allocate
     * @return Handle to the allocated resource
     */
    virtual void * alloc(size_t size) noexcept = 0;
    /**
     * @brief Releases handle and all associated memory resources, invalidates handle
     * @return false if handle cannot be release, otherwise - true.
     */
    virtual bool   free(void* handle) noexcept = 0;

 protected:
    /**
     * @brief potential issues when allocator deleted not via Release
     */
    ~IAllocator()override = default;
};

/**
 * @brief Creates the default instance of the Inference Engine interface (per plugin)
 * @return The Inference Engine interface
 */
INFERENCE_ENGINE_API(InferenceEngine::IAllocator*)CreateDefaultAllocator() noexcept;

}  // namespace InferenceEngine
