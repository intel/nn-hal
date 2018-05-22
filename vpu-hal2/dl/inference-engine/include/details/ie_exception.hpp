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
 * @brief A header file for the main Inference Engine exception
 * \file ie_exception.hpp
 */
#pragma once

#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <functional>

/**
 * @def THROW_IE_EXCEPTION
 * @brief A macro used to throw the exception with a notable description
 */
#define THROW_IE_EXCEPTION\
    throw InferenceEngine::details::InferenceEngineException(__FILE__, __LINE__)\

/**
 * @def IE_ASSERT
 * @brief Uses assert() function if NDEBUG is not defined, InferenceEngine exception otherwise
 */
#ifdef NDEBUG
    #define IE_ASSERT(EXPRESSION)\
    if (!(EXPRESSION)) throw InferenceEngine::details::InferenceEngineException(__FILE__, __LINE__)\
    << "AssertionFailed: " << #EXPRESSION
#else
#include <cassert>
#define IE_ASSERT(EXPRESSION)\
    assert((EXPRESSION))
#endif  // NDEBUG

namespace InferenceEngine {
namespace details {

/**
 * @class InferenceEngineException
 * @brief The InferenceEngineException class implements the main Inference Engine exception
 */
class InferenceEngineException : public std::exception {
    mutable std::string errorDesc;
    std::string _file;
    int _line;
    std::shared_ptr<std::stringstream> exception_stream;

public:
    /**
     * @brief A C++ std::exception API member
     * @return An exception description with a file name and file line
     */
    const char *what() const noexcept override {
        if (errorDesc.empty() && exception_stream) {
            errorDesc = exception_stream->str();
#ifndef NDEBUG
            errorDesc +=  "\n" + _file + ":" + std::to_string(_line);
#endif
        }
        return errorDesc.c_str();
    }

    /**
     * @brief A constructor. Creates an InferenceEngineException object from a specific file and line
     * @param filename File where exception has been thrown
     * @param line Line of the exception emitter
     */
    InferenceEngineException(const std::string &filename, const int line)
            : _file(filename), _line(line) {
    }

    /**
     * @brief A stream output operator to be used within exception
     * @param arg Object for serialization in the exception message
     */
    template<class T>
    InferenceEngineException &operator<<(const T &arg) {
        if (!exception_stream) {
            exception_stream.reset(new std::stringstream());
        }
        (*exception_stream) << arg;
        return *this;
    }
};
}  // namespace details
}  // namespace InferenceEngine
