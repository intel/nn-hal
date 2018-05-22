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
 * @brief The macro defines a symbol export/import mechanism for Microsoft Windows(R) OS. 
 * @file ie_api.h
 */
#pragma once

#include "details/ie_no_copy.hpp"

#if defined(_WIN32) && !defined(USE_STATIC_IE)
    #define INFERENCE_ENGINE_CDECL
    #ifdef IMPLEMENT_INFERENCE_ENGINE_API
            #define INFERENCE_ENGINE_API(type) extern "C"   __declspec(dllexport) type __cdecl
            #define INFERENCE_ENGINE_API_CPP(type)  __declspec(dllexport) type __cdecl
            #define INFERENCE_ENGINE_API_CLASS(type)        __declspec(dllexport) type
    #else
            #define INFERENCE_ENGINE_API(type) extern "C"  __declspec(dllimport) type __cdecl
            #define INFERENCE_ENGINE_API_CPP(type)  __declspec(dllimport) type __cdecl
            #define INFERENCE_ENGINE_API_CLASS(type)   __declspec(dllimport) type
    #endif
#else
#define INFERENCE_ENGINE_API(TYPE) extern "C" TYPE
#define INFERENCE_ENGINE_API_CPP(type) type
#define INFERENCE_ENGINE_API_CLASS(type)    type
#define INFERENCE_ENGINE_CDECL __attribute__((cdecl))
#endif
