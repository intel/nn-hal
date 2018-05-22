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
#include <file_utils.h>
#include "details/ie_exception.hpp"
#include <fstream>

#include <w_unistd.h>

#ifdef __MACH__
    #include <mach/clock.h>
    #include <mach/mach.h>
#endif

#if defined(WIN32) || defined(WIN64)
    // Copied from linux libc sys/stat.h:
    #define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
    #define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#endif

long long FileUtils::fileSize(const char *fileName) {
    std::ifstream in(fileName, std::ios_base::binary | std::ios_base::ate);
    return in.tellg();
}

void FileUtils::readAllFile(const std::string &file_name, void *buffer, size_t maxSize) {
    std::ifstream inputFile;

    inputFile.open(file_name, std::ios::binary | std::ios::in);
    if (!inputFile.is_open()) THROW_IE_EXCEPTION << "cannot open file " << file_name;
    if (!inputFile.read(reinterpret_cast<char *>(buffer), maxSize)) {
        inputFile.close();
        THROW_IE_EXCEPTION << "cannot read " << maxSize << " bytes from file " << file_name;
    }

    inputFile.close();
}

std::string FileUtils::folderOf(const std::string &filepath) {
    auto pos = filepath.rfind(FileSeparator);
    if (pos == std::string::npos) pos = filepath.rfind(FileSeparator2);
    if (pos == std::string::npos) return "";
    return filepath.substr(0, pos);
}

std::string FileUtils::makePath(const std::string &folder, const std::string &file) {
    if (folder.empty()) return file;
    return folder + FileSeparator + file;
}

std::string FileUtils::fileNameNoExt(const std::string &filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos) return filepath;
    return filepath.substr(0, pos);
}

std::string FileUtils::fileExt(const char *filename) {
    return fileExt(std::string(filename));
}

std::string FileUtils::fileExt(const std::string &filename) {
    auto pos = filename.rfind('.');
    if (pos == std::string::npos) return "";
    return filename.substr(pos + 1);
}

bool FileUtils::isSharedLibrary(const std::string& fileName) {
    return 0 == strncasecmp(fileExt(fileName).c_str(), SharedLibraryExt, strlen(SharedLibraryExt));
}
