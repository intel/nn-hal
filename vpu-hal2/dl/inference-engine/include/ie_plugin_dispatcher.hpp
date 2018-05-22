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
* @brief A header for a class to handle plugin loading.
* @file ie_plugin_dispatcher.hpp
*/
#pragma once

#include "ie_plugin_ptr.hpp"
#include <string>
#include <vector>

namespace InferenceEngine {
/**
* @class PluginDispatcher
* @brief This is a class to load a suitable plugin
*/
class PluginDispatcher {
public:
    /**
     * @brief A constructor
     * @param pp Vector of paths to plugin directories
     */
    explicit PluginDispatcher(const std::vector<std::string> &pp) : pluginDirs(pp) {}

    /**
    * @brief Loads a plugin from plugin directories
    * @param name Plugin name
    * @return A pointer to the plugin
    */
    virtual InferenceEnginePluginPtr getPluginByName(const std::string& name) const {
        std::stringstream err;
        for (auto &pluginPath : pluginDirs) {
            try {
                return InferenceEnginePluginPtr(make_plugin_name(pluginPath, name));
            }
            catch (const std::exception &ex) {
                err << "cannot load plugin: " << name << " from " << pluginPath << ": " << ex.what() << ", skipping\n";
            }
        }
        THROW_IE_EXCEPTION << "Plugin " << name << " cannot be loaded: " << err.str() << "\n";
    }

    /**
    * @brief Loads a plugin from directories that is suitable for the device string
    * @return A pointer to the plugin
    */
    InferenceEnginePluginPtr getPluginByDevice(const std::string& deviceName) const {
        InferenceEnginePluginPtr ptr;
        // looking for HETERO: if can find, add everything after ':' to the options of hetero plugin
        if (deviceName.find("HETERO:") == 0) {
            ptr = getSuitablePlugin(InferenceEngine::TargetDeviceInfo::fromStr("HETERO"));
            if (ptr) {
                InferenceEngine::ResponseDesc response;
                ptr->SetConfig({ { "TARGET_FALLBACK", deviceName.substr(7, deviceName.length() - 7) } }, &response);
            }
        } else {
            ptr = getSuitablePlugin(InferenceEngine::TargetDeviceInfo::fromStr(deviceName));
        }
        return ptr;
    }

    /**
    * @brief Loads a plugin from directories that is suitable for the device
    * @return A pointer to the plugin
    */
    InferenceEnginePluginPtr getSuitablePlugin(TargetDevice device) const {
        FindPluginResponse result;
        ResponseDesc desc;
        if (InferenceEngine::OK != findPlugin({ device }, result, &desc)) {
            THROW_IE_EXCEPTION << desc.msg;
        }

        std::stringstream err;
        for (std::string& name : result.names) {
            try {
                return getPluginByName(name);
            }
            catch (const std::exception &ex) {
                err << "Tried load plugin : " << name << ",  error: " << ex.what() << "\n";
            }
        }
        THROW_IE_EXCEPTION << "Cannot find plugin to use :" << err.str() << "\n";
    }

protected:
    /**
    * @brief Sets the path to the plugin
    * @param path Path to the plugin
    * @param input Plugin name
    * @return The path to the plugin
    */
    std::string make_plugin_name(const std::string &path, const std::string &input) const {
        std::string separator =
#if defined _WIN32 || defined __CYGWIN__
        "\\";
#else
        "/";
#endif
        if (path.empty())
            separator = "";
#ifdef _WIN32
        return path + separator + input + ".dll";
#elif __APPLE__
        return path + separator + "lib" + input + ".dylib";
#else
        return path + separator + "lib" + input + ".so";
#endif
    }

private:
    std::vector<std::string> pluginDirs;
};
}  // namespace InferenceEngine
