////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2017 Intel Corporation. All Rights Reserved.
//
//

#ifndef CHECK_CYCLES_HPP
#define CHECK_CYCLES_HPP

#include <exception>
#include <string>

#include "passes/pass_base.hpp"

namespace ade
{

namespace passes
{
class CycleFound : public std::exception
{
public:
    virtual const char* what() const noexcept override;
};

struct CheckCycles
{
    void operator()(const PassContext& context) const;
    static std::string name();
};
}
}

#endif // CHECK_CYCLES_HPP
