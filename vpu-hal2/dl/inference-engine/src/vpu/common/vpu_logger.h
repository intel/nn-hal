//
// INTEL CONFIDENTIAL
// Copyright 2017 Intel Corporation.
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

#include <stdint.h>
#include <stdarg.h>
#include <memory>

namespace VPU {
namespace Common {

enum LogLevel {
    eLOGNONE = 0,
    eLOGWARNING,
    eLOGINFO,
    eLOGDEBUG
};

class Logger {
public:
    Logger();

    void init(int logLevel);

    void logMessage(int msgLevel, const char* file, int line, const char* msg, ...);
    int getLogLevel();

private:
    int logLevel_;
};

typedef std::shared_ptr<Logger> LoggerPtr;

}  // namespace Common
}  // namespace VPU

#define LOG_WARNING(msg, ...) _log->logMessage(::VPU::Common::eLOGWARNING, __FILE__, __LINE__, msg, ##__VA_ARGS__)
#define LOG_INFO(msg, ...)    _log->logMessage(::VPU::Common::eLOGINFO, __FILE__, __LINE__, msg, ##__VA_ARGS__)
#define LOG_DEBUG(msg, ...)   _log->logMessage(::VPU::Common::eLOGDEBUG, __FILE__, __LINE__, msg, ##__VA_ARGS__)
