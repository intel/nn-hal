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

/**
 * @brief POSIX compatible loader for a shared object
 * @file lin_shared_object_loader.h
 */
#pragma once

#include <dlfcn.h>

#include "../../ie_api.h"
#include "../ie_exception.hpp"

/**
 * @class SharedObjectLoader
 * @brief This class provides an OS shared module abstraction
 */
class SharedObjectLoader {
private:
    void *shared_object = nullptr;

public:
    /**
     * @brief Loads a library with the name specified. The library is loaded according to
     *        the POSIX rules for dlopen
     * @param pluginName Full or relative path to the plugin library
     */
    explicit SharedObjectLoader(const char* pluginName) {
        shared_object = dlopen(pluginName, RTLD_LAZY);

        if (shared_object == nullptr)
            THROW_IE_EXCEPTION << "Cannot load library '" << pluginName << "': " << dlerror();
    }
    ~SharedObjectLoader() noexcept(false) {
        if (0 != dlclose(shared_object)) {
            THROW_IE_EXCEPTION << "dlclose failed: " << dlerror();
        }
    }

    /**
     * @brief Searches for a function symbol in the loaded module
     * @param symbolName Name of the function to find
     * @return A pointer to the function if found
     * @throws InferenceEngineException if the function is not found
     */
    void *get_symbol(const char* symbolName) const {
        void * procAddr = nullptr;

        procAddr = dlsym(shared_object, symbolName);
        if (procAddr == nullptr)
            THROW_IE_EXCEPTION << "dlSym cannot locate method '" << symbolName << "': " << dlerror();
        return procAddr;
    }
};
