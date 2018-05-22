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

#include "vpu_logger.h"
#include "details/ie_exception.hpp"

namespace {

const char* COLOR_RED = "\e[1;31m";
const char* COLOR_GRN = "\e[1;32m";
const char* COLOR_YEL = "\e[1;33m";
const char* COLOR_BLU = "\e[1;34m";
const char* COLOR_MAG = "\e[1;35m";
const char* COLOR_CYN = "\e[1;36m";
const char* COLOR_END = "\e[0m";

const char* COLOR_WARNING = COLOR_RED;
const char* COLOR_INFO    = COLOR_GRN;
const char* COLOR_DEBUG   = COLOR_YEL;

}  // namespace

VPU::Common::Logger::Logger() : logLevel_(eLOGNONE) {
}

void VPU::Common::Logger::init(int logLevel) {
    if (logLevel < eLOGNONE || logLevel > eLOGDEBUG) {
        THROW_IE_EXCEPTION << "Incorrect log level " << logLevel;
    }

    logLevel_ = logLevel;
}

void VPU::Common::Logger::logMessage(int msgLevel, const char* file, int line, const char* msg, ...) {
    if (msgLevel > logLevel_) {
        return;
    }

    va_list args;
    va_start(args, msg);

    char buf[1024];

    switch (msgLevel) {
    case eLOGWARNING:
        printf("%s", COLOR_WARNING);
        vprintf(msg, args);
        printf("%s", COLOR_END);
        break;
    case eLOGINFO:
        printf("%s", COLOR_INFO);
        vprintf(msg, args);
        printf("%s", COLOR_END);
        break;
    case eLOGDEBUG:
        vsnprintf(buf, sizeof(buf), msg, args);

        printf("%s", COLOR_DEBUG);
        printf("%s at %s:%d", buf, file, line);
        printf("%s", COLOR_END);
        break;
    default:
        vprintf(msg, args);
        break;
    }

    va_end(args);

    printf("\n");
    fflush(stdout);
}

int VPU::Common::Logger::getLogLevel() {
    return logLevel_;
}
