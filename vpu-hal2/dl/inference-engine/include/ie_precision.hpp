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


#pragma once
#include <unordered_map>
#include <string>
#include "details/ie_exception.hpp"

namespace InferenceEngine {

/**
 * @class Precision
 * @brief This class holds precision value and provides precision related operations
 */
class Precision {
 public:
    enum ePrecision : uint8_t {
        UNSPECIFIED = 255,
        MIXED = 0,
        FP32 = 10,
        FP16 = 11,
        Q78 = 20,
        I16 = 30,
        U8 = 40,
        I8 = 50,
        U16 = 60,
        I32 = 70,
        CUSTOM = 80  // custom precision has it's own name and size of elements
    };

 private:
    struct PrecisionInfo {
        /**
         * @brief size of unerlined element
         */
        size_t bitsSize = 0;
        /**
         * null terminated pointer to precision name
         */
        const char *name = "UNSPECIFIED";
        bool isFloat     = false;
        ePrecision value = Precision::UNSPECIFIED;
    };
    PrecisionInfo precisionInfo;

 public:
    Precision()  = default;

    Precision(const Precision::ePrecision  value) {
        precisionInfo = getPrecisionInfo(value);
    }

    /**
     * @brief custom precision constructor
     * @param byteSize - size of elements
     * @param name - optional name string, used in serialisation
     */
    explicit Precision(size_t bitsSize, const char * name = nullptr) {
        if (bitsSize == 0) {
            THROW_IE_EXCEPTION << "Precision with 0 elements size not supported";
        }
        precisionInfo.bitsSize = bitsSize;
        if (name == nullptr) {
            precisionInfo.name = "CUSTOM";
        } else {
            precisionInfo.name = name;
        }
        precisionInfo.value = CUSTOM;
    }

    /**
    * @brief creates custom precision with specific underlined type
    */
#ifdef AKS
    template <class T>
    static Precision fromType(const char * typeName) {
        return Precision(8 * sizeof(T), typeName);
    }
#else
    template <class T>
    static Precision fromType(const char * typeName = nullptr) {
        return Precision(8 * sizeof(T), typeName == nullptr ? typeid(T).name() : typeName);
    }
#endif
    bool operator == (const Precision  & p) const noexcept  {
        return precisionInfo.value == p &&
            precisionInfo.bitsSize == p.precisionInfo.bitsSize &&
            areSameStrings(precisionInfo.name, p.precisionInfo.name);
    }

    bool operator == (const ePrecision  p) const noexcept  {
        return precisionInfo.value == p;
    }

    bool operator != (const ePrecision   p) const noexcept  {
        return precisionInfo.value != p;
    }

    Precision & operator = (const ePrecision p) noexcept {
        precisionInfo = getPrecisionInfo(p);
        return *this;
    }

    explicit operator bool() const noexcept {
        return precisionInfo.value != UNSPECIFIED;
    }

    bool operator !() const noexcept {
        return precisionInfo.value == UNSPECIFIED;
    }

    operator Precision::ePrecision  () const noexcept {
        return precisionInfo.value;
    }

    const char *name() const noexcept {
        return precisionInfo.name;
    }

#define PRECISION_NAME(s) {#s, s}
    static Precision FromStr(const std::string &str) {
        static std::unordered_map<std::string, ePrecision > names = {
            PRECISION_NAME(Q78),
            PRECISION_NAME(U8),
            PRECISION_NAME(I8),
            PRECISION_NAME(I16),
            PRECISION_NAME(I32),
            PRECISION_NAME(U16),
            PRECISION_NAME(FP32),
            PRECISION_NAME(FP16),
            PRECISION_NAME(MIXED),
        };
        auto i = names.find(str);
        return i == names.end() ? Precision() : Precision(i->second);
    }
#undef PRECISION_NAME
    /** @brief size in bytes of single element of that precision
     * @deprecated : size of precision will be report in bits in future releases
     */
    size_t size() const {
        if (precisionInfo.bitsSize == 0) {
            THROW_IE_EXCEPTION << " cannot estimate element if precision is " << precisionInfo.name;
        }
        return precisionInfo.bitsSize >> 3;
    }

    bool is_float() const {
        return precisionInfo.isFloat;
    }

 protected:
    template<Precision::ePrecision precision>
    static PrecisionInfo makePrecisionInfo(const char * name);

    static bool areSameStrings(const char *l, const char *r) {
        if (l == r)
            return true;

        if (l == nullptr || r == nullptr)
            return false;

        for (; *l && *r; l++, r++) {
            if (*l != *r) return false;
        }
        return *l == *r;
    }

    static PrecisionInfo getPrecisionInfo(ePrecision v) {
#define CASE(x) case x: return makePrecisionInfo<x>(#x);
        switch (v) {
            CASE(FP32);
            CASE(FP16);
            CASE(I16);
            CASE(I32);
            CASE(U16);
            CASE(U8);
            CASE(I8);
            CASE(Q78);
            CASE(MIXED);
            default : return makePrecisionInfo<UNSPECIFIED>("UNSPECIFIED");
#undef CASE
        }
    }
};

template<Precision::ePrecision p>
struct PrecisionTrait {
};

template<>
struct PrecisionTrait<Precision::FP32> {
    using value_type = float;
};
template<>
struct PrecisionTrait<Precision::FP16> {
    using value_type = uint16_t;
};
template<>
struct PrecisionTrait<Precision::Q78> {
    using value_type = uint16_t;
};
template<>
struct PrecisionTrait<Precision::I16> {
    using value_type = int16_t;
};
template<>
struct PrecisionTrait<Precision::U16> {
    using value_type = uint16_t;
};
template<>
struct PrecisionTrait<Precision::U8> {
    using value_type = uint8_t;
};
template<>
struct PrecisionTrait<Precision::I8> {
    using value_type = int8_t;
};
template<>
struct PrecisionTrait<Precision::I32> {
    using value_type = int32_t;
};

template<class T>
inline uint8_t type_size_or_zero() {
    return sizeof(T);
}

template<>
struct PrecisionTrait<Precision::UNSPECIFIED> {
    using value_type = void;
};

template<>
struct PrecisionTrait<Precision::MIXED> : PrecisionTrait<Precision::UNSPECIFIED>{
};

template<>
inline uint8_t type_size_or_zero<void>() {
    return 0;
}

/**
 * support for FP16
 */
template<Precision::ePrecision T>
inline typename std::enable_if<std::is_same<
    std::integral_constant<Precision::ePrecision, Precision::FP16>,
    std::integral_constant<Precision::ePrecision, T>>::value, bool>::type is_floating() {
    return true;
}

template<Precision::ePrecision T>
inline typename std::enable_if<!std::is_same<
    std::integral_constant<Precision::ePrecision, Precision::FP16>,
    std::integral_constant<Precision::ePrecision, T>>::value, bool>::type is_floating() {
    return std::is_floating_point<typename PrecisionTrait<T>::value_type>::value;
}

template<Precision::ePrecision precision>
inline Precision::PrecisionInfo Precision::makePrecisionInfo(const char *name) {
    Precision::PrecisionInfo info;
    info.name = name;
    info.bitsSize = 8 * type_size_or_zero<typename PrecisionTrait<precision>::value_type>();
    info.isFloat = is_floating<precision>();
    info.value = precision;
    return info;
}

inline std::ostream & operator << (std::ostream &out, const InferenceEngine::Precision & p) {
    return out << p.name();
}

inline std::ostream & operator << (std::ostream &out, const InferenceEngine::Precision::ePrecision & p) {
    return out << Precision(p).name();
}

}  // namespace InferenceEngine
