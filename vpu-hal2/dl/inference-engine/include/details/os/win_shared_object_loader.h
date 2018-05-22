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
 * @brief WINAPI based dll loader
 * @file win_shared_object_loader.h
 */
#pragma once

#include "../../ie_api.h"
#include "../ie_exception.hpp"

// Avoidance of Windows.h to include winsock library.
#define _WINSOCKAPI_
// Avoidance of Windows.h to define min/max.
#define NOMINMAX
#include <windows.h>
#include <direct.h>

/**
 * @class SharedObjectLoader
 * @brief This class provides an OS shared module abstraction
 */

class SharedObjectLoader {
private:
    HMODULE shared_object;

 public:
    /**
     * @brief Loads a library with the name specified. The library is loaded according to the
     *        WinAPI LoadLibrary rules
     * @param pluginName Full or relative path to the plugin library
     */
    explicit SharedObjectLoader(const char* pluginName) {
        char cwd[1024];
        shared_object = LoadLibrary(pluginName);
        if (!shared_object) {
            THROW_IE_EXCEPTION << "Cannot load library '"
                << pluginName << "': "
                << GetLastError()
                << " from cwd: " << _getcwd(cwd, 1024);
        }
    }
    ~SharedObjectLoader() {
        FreeLibrary(shared_object);
    }

    /**
     * @brief Searches for a function symbol in the loaded module
     * @param symbolName Name of function to find
     * @return A pointer to the function if found
     * @throws InferenceEngineException if the function is not found
     */
    void *get_symbol(const char* symbolName) const {
        if (!shared_object) {
            THROW_IE_EXCEPTION << "Cannot get '" << symbolName << "' content from unknown library!";
        }
        auto procAddr = reinterpret_cast<void*>(GetProcAddress(shared_object, symbolName));
        if (procAddr == nullptr)
            THROW_IE_EXCEPTION << "GetProcAddress cannot locate method '" << symbolName << "': " << GetLastError();

        return procAddr;
    }
};
