// SPDX-FileCopyrightText: 2019 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TENSORRT_COMMON_H
#define TENSORRT_COMMON_H

#include <fstream>
#include <memory>
#include <numeric>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include "logger.hpp"

using namespace nvinfer1;
using namespace plugin;

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

#if (!defined(__ANDROID__) && defined(__aarch64__)) || defined(__QNX__)
#define ENABLE_DLA_API 1
#endif

#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << std::endl;                                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#define CHECK_RETURN_W_MSG(status, val, errMsg)                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(status))                                                                                                 \
        {                                                                                                              \
            std::cerr << errMsg << " Error in " << __FILE__ << ", function " << FN_NAME << "(), line " << __LINE__     \
                      << std::endl;                                                                                    \
            return val;                                                                                                \
        }                                                                                                              \
    } while (0)

#define CHECK_RETURN(status, val) CHECK_RETURN_W_MSG(status, val, "")

#define OBJ_GUARD(A) std::unique_ptr<A, void (*)(A * t)>

template <typename T, typename T_>
OBJ_GUARD(T)
makeObjGuard(T_* t)
{
    CHECK(!(std::is_base_of<T, T_>::value || std::is_same<T, T_>::value));
    auto deleter = [](T* t) { t->destroy(); };
    return std::unique_ptr<T, decltype(deleter)>{static_cast<T*>(t), deleter};
}

namespace samplesCommon
{

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

inline void print_version()
{
    std::cout << "  TensorRT version: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH
              << "." << NV_TENSORRT_BUILD << std::endl;
}

inline void enableDLA(IBuilder* builder, IBuilderConfig* config, int useDLACore, bool allowGPUFallback = true)
{
    if (useDLACore >= 0)
    {
        if (builder->getNbDLACores() == 0)
        {
            std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores"
                      << std::endl;
            assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
        }
        if (allowGPUFallback)
        {
            config->setFlag(BuilderFlag::kGPU_FALLBACK);
        }
        if (!config->getFlag(BuilderFlag::kINT8))
        {
            // User has not requested INT8 Mode.
            // By default run in FP16 mode. FP32 mode is not permitted.
            config->setFlag(BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(DeviceType::kDLA);
        config->setDLACore(useDLACore);
        config->setFlag(BuilderFlag::kSTRICT_TYPES);
    }
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

template <typename A, typename B>
inline A divUp(A x, B n)
{
    return (x + n - 1) / n;
}

} // namespace samplesCommon

inline std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& dims)
{
    os << "(";
    for (int i = 0; i < dims.nbDims; ++i)
    {
        os << (i ? ", " : "") << dims.d[i];
    }
    return os << ")";
}

#endif // TENSORRT_COMMON_H

