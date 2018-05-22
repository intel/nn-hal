////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2017 Intel Corporation. All Rights Reserved.
//
//

#ifndef EXECUTABLE_HPP
#define EXECUTABLE_HPP

namespace util
{
    class any;
}

namespace ade
{
class Executable
{
public:
    virtual ~Executable() = default;
    virtual void run() = 0;
    virtual void run(util::any &opaque) = 0;      // WARNING: opaque may be accessed from various threads.

    virtual void runAsync() = 0;
    virtual void runAsync(util::any &opaque) = 0; // WARNING: opaque may be accessed from various threads.

    virtual void wait() = 0;
};
}

#endif // EXECUTABLE_HPP
