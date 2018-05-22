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
* @brief This is a wrapper class for handling plugin instantiation and releasing resources
* @file ie_plugin_ptr.hpp
*/
#pragma once

#include <memory>
#include "ie_so_loader.h"
#include "ie_common.h"
#include "ie_plugin.hpp"
#include "details/ie_exception.hpp"
#include "details/ie_no_release.hpp"
#include <string>
#include <cassert>

namespace InferenceEngine {
namespace details {
/**
* @class SymbolLoader
* @brief This class is a C++ helper to load a symbol from a library and create its instance
*/
class SymbolLoader {
private:
    std::shared_ptr<SharedObjectLoader> _so_loader;

public:
    /**
    * @brief The main constructor
    * @param loader Library to load from
    */
    explicit SymbolLoader(std::shared_ptr<SharedObjectLoader> loader) : _so_loader(loader) {}

    /**
    * @brief Calls a function from the library that creates an object and returns StatusCode
    * @param name Name of function to load object with
    * @return If StatusCode provided by function is OK then returns the loaded object. Throws an exception otherwise
    */
    template<class T>
    T* instantiateSymbol(const std::string& name) const {
        T* instance = nullptr;
        ResponseDesc desc;
        StatusCode sts = bind_function<StatusCode(T*&, ResponseDesc*)>(name)(instance, &desc);
        if (sts != OK) {
            THROW_IE_EXCEPTION << desc.msg;
        }
        return instance;
    }

    /**
    * @brief Loads function from the library and returns a pointer to it
    * @param functionName Name of function to load
    * @return The loaded function
    */
    template<class T>
    std::function<T> bind_function(const std::string &functionName) const {
        std::function<T> ptr(reinterpret_cast<T *>(_so_loader->get_symbol(functionName.c_str())));
        return ptr;
    }
};

/**
* @class SOCreatorTrait
* @brief This class is a trait class that provides a creator with a function name corresponding to the templated class parameter
*/
template<class T>
class SOCreatorTrait {};

/**
* @class SOPointer
* @brief This class instantiate object using shared library
*/
template <class T>
class SOPointer {
public:
    /**
    * @brief Default constructor
    */
    SOPointer() = default;

    /**
    * @brief The main constructor
    * @param name Name of a shared library file
    */
    explicit SOPointer(const std::string &name)
        : _so_loader(new SharedObjectLoader(name.c_str()))
        , _pointedObj(details::shared_from_irelease(
            SymbolLoader(_so_loader).instantiateSymbol<T>(SOCreatorTrait<T>::name))) {
    }

    /**
    * @brief Standard pointer operator
    * @return underlined interface with disabled Release method
    */
    details::NoReleaseOn<T>* operator->() const noexcept {
        return reinterpret_cast<details::NoReleaseOn<T>*>(_pointedObj.get());
    }

    /**
    * @brief Standard dereference operator
    * @return underlined interface with disabled Release method
    */
    details::NoReleaseOn<T>* operator*() const noexcept {
        return this->operator->();
    }

    explicit operator bool() const noexcept {
        assert((nullptr == _so_loader) == ((nullptr == _pointedObj)));
        return nullptr != _so_loader;
    }

    friend bool operator == (std::nullptr_t, const SOPointer& ptr) noexcept {
        return !ptr;
    }
    friend bool operator == (const SOPointer& ptr, std::nullptr_t) noexcept {
        return !ptr;
    }
    friend bool operator != (std::nullptr_t, const SOPointer& ptr) noexcept {
        return static_cast<bool>(ptr);
    }
    friend bool operator != (const SOPointer& ptr, std::nullptr_t) noexcept {
        return static_cast<bool>(ptr);
    }

protected:
    /**
     * @brief Gets a smart pointer to the DLL
     */
    std::shared_ptr<SharedObjectLoader> _so_loader;
    /**
     * @brief Gets a smart pointer to the custom object
     */
    std::shared_ptr<T> _pointedObj;
};

}  // namespace details



/**
 * @brief Creates a special shared_pointer wrapper for the given type from a specific shared module
 * @param name Name of the shared library file
 */
template <class T>
inline std::shared_ptr<T> make_so_pointer(const std::string & name) {
    throw std::logic_error("not implemented");
}

}  // namespace InferenceEngine
