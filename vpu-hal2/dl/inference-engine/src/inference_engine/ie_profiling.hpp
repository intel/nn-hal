#pragma once

#include <string>
#include <unordered_map>
#include <mutex>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <deque>
#include <chrono>

#if ENABLE_PROFILING_ITT
#include <ittnotify.h>
#endif

namespace InferenceEngine {

class TimeResultsMap {
protected:
    std::unordered_map<std::string, std::deque<double> > m_map;
    std::mutex m_lock;

public:
    void add(const  std::string& name, double val) {
        #if ENABLE_PROFILING_RAW
        m_lock.lock();
        m_map[name].push_back(val);
        m_lock.unlock();
        #endif
    }
    ~TimeResultsMap() {
        for (auto && iter : m_map) {
            const size_t num = iter.second.size();
            double valSum = 0, valMin = DBL_MAX, valMax = -DBL_MAX, logSum = 0;
            int logCount = 0;
            for (size_t i = 0; i < num; i++) {
                double_t val = iter.second[i];
                if (val > 0) {
                    logCount++;
                    logSum += std::log(val);
                }
                valSum += val;
                valMin = std::fmin(val, valMin);
                valMax = std::fmax(val, valMax);
            }

            std::cout << std::setw(20) << iter.first << " collected by " << std::setw(8) << num << " samples, ";
            std::cout << "mean " << std::setw(12) << (valSum / num)/1000000 << " ms, ";
            std::cout << "geomean " << std::setw(12) << (logCount ? std::exp(logSum / logCount) : 0)/1000000 << " ms, ";
            std::cout << "min " << std::setw(12) << valMin/1000000 << " ms, ";
            std::cout << "max " << std::setw(12) << valMax/1000000 << " ms" << std::endl;
        }
    }
};

class TimeSampler {
public:
    TimeSampler(const char *pName) : m_name(pName) {
        #if ENABLE_PROFILING_ITT
        m_handle = __itt_string_handle_create(m_name.c_str());
        #endif
    }

    void start() {
        #if ENABLE_PROFILING_ITT
        __itt_task_begin(InferenceEngine::TimeSampler::globalIEDomain(), __itt_null, __itt_null, m_handle);
        #endif
        #if ENABLE_PROFILING_RAW
        m_startTime = std::chrono::high_resolution_clock::now();
        #endif
    }

    void stop() {
        #if ENABLE_PROFILING_ITT
        __itt_task_end(InferenceEngine::TimeSampler::globalIEDomain());
        #endif
        #if ENABLE_PROFILING_RAW
        double val =  std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - m_startTime).count();
        globalCountersMap().add(m_name, val);
        #endif
    }
    const std::string m_name;
    std::chrono::high_resolution_clock::time_point m_startTime;

    #if ENABLE_PROFILING_ITT
    __itt_string_handle* m_handle;
    __itt_domain* globalIEDomain() {
        static __itt_domain* ittDomain = __itt_domain_create("InferenceEngine");
        return ittDomain;
    }
    #endif
    TimeResultsMap& globalCountersMap() {
        static TimeResultsMap map;
        return map;
    }
};

class ProfilingScopeAuto {
    InferenceEngine::TimeSampler sampler;

public:
    ProfilingScopeAuto(const char* pName) : sampler(pName) {
        sampler.start();
    }

    ~ProfilingScopeAuto() {
        sampler.stop();
    }
};

#if ENABLE_PROFILING_ITT || ENABLE_PROFILING_RAW
    // for declaration as class members
#define IE_PROFILING_DECLARE(NAME)    InferenceEngine::TimeSampler __ie_profiling_##NAME;
    // use in the constructor init list
#define IE_PROFILING_CONSTRUCT(NAME)  __ie_profiling_##NAME(#NAME)
    // for use as explicitly defined local variables
#define IE_PROFILING_DEFINE(NAME)    InferenceEngine::TimeSampler __ie_profiling_##NAME(#NAME);
#define IE_PROFILING_BEGIN(NAME)     __ie_profiling_##NAME.start();
#define IE_PROFILING_END(NAME)       __ie_profiling_##NAME.stop();
    // for use as auto local variables (that capture perf in the destructor)
#define IE_PROFILING_AUTO_SCOPE(NAME) InferenceEngine::ProfilingScopeAuto __ie_profiling_auto_##NAME(#NAME);
#define IE_PROFILING_AUTO_SCOPE_STRING(STR) InferenceEngine::ProfilingScopeAuto __ie_profiling_auto(STR);
#else
    #define IE_PROFILING_DECLARE(NAME)
    #define IE_PROFILING_CONSTRUCT(NAME)
    #define IE_PROFILING_DEFINE(NAME)
    #define IE_PROFILING_BEGIN(NAME)
    #define IE_PROFILING_END(NAME)
    #define IE_PROFILING_AUTO_SCOPE(NAME)
    #define IE_PROFILING_AUTO_SCOPE_STRING(STR)
#endif
}  // namespace InferenceEngine
