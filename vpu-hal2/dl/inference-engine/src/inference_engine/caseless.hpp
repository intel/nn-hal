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
#include <functional>
#include <unordered_map>
#include <map>
#include <set>
#include <cctype>

/**
 * @brief provides case-less comparison for stl algorithms
 * @tparam Key type, usually std::string
 */
template<class Key>
class CaselessLess : public std::binary_function<Key, Key, bool> {
 public:
    bool operator () (const Key & a, const Key & b) const noexcept {
        return std::lexicographical_compare(std::begin(a),
                          std::end(a),
                          std::begin(b),
                          std::end(b),
                          [](const char&cha, const char&chb) {
                              return std::tolower(cha) < std::tolower(chb);
                          });
    }
};

/**
 * provides caseless eq for stl algorithms
 * @tparam Key
 */
template<class Key>
class CaselessEq : public std::binary_function<Key, Key, bool> {
 public:
    bool operator () (const Key & a, const Key & b) const noexcept {
        return a.size() == b.size() &&
            std::equal(std::begin(a),
                       std::end(a),
                       std::begin(b),
                       [](const char&cha, const char&chb) {
                           return std::tolower(cha) == std::tolower(chb);
                       });
    }
};

/**
 * To hash caseless
 */
template<class T>
class CaselessHash : public std::hash<T> {
 public:
    size_t operator()(T __val) const noexcept {
          T lc;
          std::transform(std::begin(__val), std::end(__val), std::back_inserter(lc), [](typename T::value_type ch) {
              return std::tolower(ch);
          });
          return std::hash<T>()(lc);
      }
};

template <class Key, class Value>
using caseless_unordered_map = std::unordered_map<Key, Value, CaselessHash<Key>, CaselessEq<Key>>;

template <class Key, class Value>
using caseless_map = std::map<Key, Value, CaselessLess<Key>>;

template <class Key>
using caseless_set = std::set<Key, CaselessLess<Key>>;