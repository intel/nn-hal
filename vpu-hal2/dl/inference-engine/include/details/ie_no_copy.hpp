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
 * @brief header file for no_copy class
 * @file ie_no_copy.hpp
 */
#pragma once

namespace InferenceEngine {
namespace details {
/**
 * @class no_copy
 * @brief This class is used for objects returned from the shared library factory to prevent copying
 */
class no_copy {
protected:
    /**
     * @brief A default constructor
     */
    no_copy() = default;

    /**
     * @brief A default destructor
     */
    virtual ~no_copy() = default;

    /**
     * @brief A removed copy constructor
     */
    no_copy(no_copy const &) = delete;

    /**
     * @brief A removed assign operator
     */
    no_copy &operator=(no_copy const &) = delete;

    /**
     * @brief A removed move constructor
     */
    no_copy(no_copy &&) = delete;

    /**
     * @brief A removed move operator
     */
    no_copy &operator=(no_copy &&) = delete;
};
}  // namespace details
}  // namespace InferenceEngine
