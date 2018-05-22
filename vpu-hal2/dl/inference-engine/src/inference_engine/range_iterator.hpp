//
// INTEL CONFIDENTIAL
// Copyright 2016 Intel Corporation.
//
// The source code contained or described herein and all documents
// related to the source code ("Material") are owned by Intel Corporation
// or its suppliers or licensors. Title to the Material remains with
// Intel Corporation or its suppliers and licensors. The Material may
// contain trade secrets and proprietary and confidential information
// of Intel Corporation and its suppliers and licensors, and is protected
// by worldwide copyright and trade secret laws and treaty provisions.
// No part of the Material may be used, copied, reproduced, modified,
// published, uploaded, posted, transmitted, distributed, or disclosed
// in any way without Intel's prior express written permission.
//
// No license under any patent, copyright, trade secret or other
// intellectual property right is granted to or conferred upon you by
// disclosure or delivery of the Materials, either expressly, by implication,
// inducement, estoppel or otherwise. Any license under such intellectual
// property rights must be express and approved by Intel in writing.
//
// Include any supplier copyright notices as supplier requires Intel to use.
//
// Include supplier trademarks or logos as supplier requires Intel to use,
// preceded by an asterisk. An asterisked footnote can be added as follows:
// *Third Party trademarks are the property of their respective owners.
//
// Unless otherwise agreed by Intel in writing, you may not remove or alter
// this notice or any other notice embedded in Materials by Intel or Intel's
// suppliers or licensors in any way.
//

#pragma once

#include <algorithm>

namespace InferenceEngine {

/**
 * @Brief iterator for accesing standard c-style null terminated strings withing c++ algorithms
 * @tparam Char
 */
template<typename Char>
struct null_terminated_range_iterator : public std::iterator<std::forward_iterator_tag, Char> {
 public:
    null_terminated_range_iterator() = delete;

    // make a non-end iterator (well, unless you pass nullptr ;)
    explicit null_terminated_range_iterator(Char *ptr) : ptr(ptr) {}

    bool operator != (null_terminated_range_iterator const &that) const {
        // iterators are equal if they point to the same location
        return !(operator==(that));
    }

    bool operator == (null_terminated_range_iterator const &that) const {
        // iterators are equal if they point to the same location
        return ptr == that.ptr
            // or if they are both end iterators
            || (is_end() && that.is_end());
    }

    null_terminated_range_iterator<Char> &operator++() {
        get_accessor()++;
        return *this;
    }

    null_terminated_range_iterator<Char> &operator++(int) {
        return this->operator++();
    }

    Char &operator*() {
        return *get_accessor();
    }

 protected:
    Char *& get_accessor()  {
        if (ptr == nullptr) {
            throw std::logic_error("null_terminated_range_iterator dereference: pointer is zero");
        }
        return ptr;
    }
    bool is_end() const {
        // end iterators can be created by the default ctor
        return !ptr
            // or by advancing until a null character
            || !*ptr;
    }

    Char *ptr;
};

template<typename Char>
struct null_terminated_range_iterator_end : public null_terminated_range_iterator<Char> {
 public:
    // make an end iterator
    null_terminated_range_iterator_end() :  null_terminated_range_iterator<Char>(nullptr) {
        null_terminated_range_iterator<Char>::ptr = nullptr;
    }
};


inline null_terminated_range_iterator<const char> null_terminated_string(const char *a) {
    return null_terminated_range_iterator<const char>(a);
}

inline null_terminated_range_iterator<const char> null_terminated_string_end() {
    return null_terminated_range_iterator_end<const char>();
}

}  // namespace InferenceEngine
