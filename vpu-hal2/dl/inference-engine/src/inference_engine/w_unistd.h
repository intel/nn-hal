// Copyright (c) 2017 Intel Corporation
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

#pragma once

#if defined(_WIN32)

#include <winsock2.h>
#include <windows.h>
#include <stdlib.h>
#include <process.h>
#include <direct.h>
#include <io.h>

#define strncasecmp _strnicmp
#define getcwd _getcwd
#define fileno _fileno

#define SecuredGetEnv GetEnvironmentVariableA

static void usleep(long microSecs) { Sleep(microSecs / 1000); }
#else

#include <unistd.h>
#include <cstdlib>
#include <string.h>

static inline int SecuredGetEnv(const char *envName, char *buf, int bufLen) {
    char *pe = getenv(envName);
    if (!pe) return 0;
    strncpy(buf, pe, bufLen - 1);
    buf[bufLen - 1] = 0;
    return strlen(buf);
}

#endif


