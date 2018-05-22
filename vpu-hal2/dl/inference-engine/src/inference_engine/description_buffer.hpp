/*
 * INTEL CONFIDENTIAL
 * Copyright 2016 Intel Corporation.
 *
 * The source code contained or described herein and all documents
 * related to the source code ("Material") are owned by Intel Corporation
 * or its suppliers or licensors. Title to the Material remains with
 * Intel Corporation or its suppliers and licensors. The Material may
 * contain trade secrets and proprietary and confidential information
 * of Intel Corporation and its suppliers and licensors, and is protected
 * by worldwide copyright and trade secret laws and treaty provisions.
 * No part of the Material may be used, copied, reproduced, modified,
 * published, uploaded, posted, transmitted, distributed, or disclosed
 * in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other
 * intellectual property right is granted to or conferred upon you by
 * disclosure or delivery of the Materials, either expressly, by implication,
 * inducement, estoppel or otherwise. Any license under such intellectual
 * property rights must be express and approved by Intel in writing.
 *
 * Include any supplier copyright notices as supplier requires Intel to use.
 *
 * Include supplier trademarks or logos as supplier requires Intel to use,
 * preceded by an asterisk. An asterisked footnote can be added as follows:
 * *Third Party trademarks are the property of their respective owners.
 *
 * Unless otherwise agreed by Intel in writing, you may not remove or alter
 * this notice or any other notice embedded in Materials by Intel or Intel's
 * suppliers or licensors in any way.
 */

#pragma once

#include <ostream>
#include <memory>
#include "ie_common.h"
#include <string>

namespace InferenceEngine {
struct DescriptionBuffer : public std::basic_streambuf<char, std::char_traits<char> > {
    std::unique_ptr<std::ostream> stream;
    StatusCode err = GENERAL_ERROR;

    DescriptionBuffer(StatusCode err, ResponseDesc *desc)
            : err(err) {
        init(desc);
    }

    explicit DescriptionBuffer(StatusCode err)
        : err(err) {
    }

    explicit DescriptionBuffer(ResponseDesc *desc) {
        init(desc);
    }

    DescriptionBuffer(char *pBuffer, size_t len) {
        init(pBuffer, len);
    }

    DescriptionBuffer(StatusCode err, char *pBuffer, size_t len)
            : err(err) {
        init(pBuffer, len);
    }


    template<class T>
    DescriptionBuffer &operator<<(const T &obj) {
        if (!stream)
            return *this;
        (*stream.get()) << obj;

        return *this;
    }

    operator StatusCode() const {
        if (stream)
            stream->flush();
        return err;
    }

private:
    void init(ResponseDesc *desc) {
        if (desc == nullptr)
            return;
        init(desc->msg, sizeof(desc->msg) / sizeof(desc->msg[0]));
    }

    void init(char *ptr, size_t len) {
        if (nullptr != ptr) {
            // set the "put" pointer the start of the buffer and record it's length.
            setp(ptr, ptr + len - 1);
        }
        stream.reset(new std::ostream(this));

        if (nullptr != ptr) {
            (*stream.get()) << ptr;
            ptr[len - 1] = 0;
        }
    }
};
}  // namespace InferenceEngine
