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
 * @brief A header file for the BlobIterator class
 * @file ie_blob_iterator.hpp
 */

#include "ie_locked_memory.hpp"
#include <utility>

namespace InferenceEngine {
namespace details {
/**
 * @class BlobIterator
 * @brief This class provides range loops support for TBlob objects
 */
template<class T>
class BlobIterator {
    LockedMemory<T> _mem;
    size_t _offset;

public:
    /**
     * @brief A move constructor to create a BlobIterator instance from a LockedMemory instance.
     * Explicitly rejects implicit conversions.
     * @param lk Rvalue of the memory instance to move from
     * @param offset Size of offset in memory
     */
    explicit BlobIterator(LockedMemory<T> &&lk, size_t offset = 0)
            : _mem(std::move(lk)), _offset(offset) {
    }

    /**
     * @brief Increments an offset of the current BlobIterator instance
     * @return The current BlobIterator instance
     */
    BlobIterator &operator++() {
        _offset++;
        return *this;
    }

    /**
     * @brief An overloaded postfix incrementation operator
     * Implementation does not follow std interface since only move semantics is used
     */
    void operator++(int) {
        _offset++;
    }

    /**
     * @brief Checks if the given iterator is not equal to the current one
     * @param that Iterator to compare with
     * @return true if the given iterator is not equal to the current one, false - otherwise
     */
    bool operator!=(const BlobIterator &that) const {
        return !operator==(that);
    }

    /**
     * @brief Gets a value by the pointer to the current iterator
     * @return The value stored in memory for the current offset value
     */
    const T &operator*() const {
        return *(_mem.template as<const T *>() + _offset);
    }

    /**
     * @brief Gets a value by the pointer to the current iterator
     * @return The value stored in memory for the current offset value
     */
    T &operator*() {
        return *(_mem.template as<T *>() + _offset);
    }
    /**
     * @brief Compares the given iterator with the current one
     * @param that Iterator to compare with
     * @return true if the given iterator is equal to the current one, false - otherwise
     */
    bool operator==(const BlobIterator &that) const {
        return &operator*() == &that.operator*();
    }
};
}  // namespace details
}  // namespace InferenceEngine
