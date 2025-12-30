// ImageProcess_c.h
// MNN
//
// Created by MNN on 2018/09/19.
// Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_ImageProcess_c_h
#define MNN_ImageProcess_c_h

#include <MNN/HalideRuntime.h>
#include <ErrorCode_c.h>
#include <Tensor_c.h>
#include <Matrix_c.h>

// 导出宏：Windows下导出，其他平台兼容
#ifdef MNN_C_EXPORTS
    // 仅当编译 libmnn.dll 时定义 MNN_C_EXPORTS，此时用 dllexport
    #define MNN_C_API __declspec(dllexport)
#else
    #define MNN_C_API
#endif

// Define bool type for C compatibility
#ifdef __cplusplus
typedef bool MNN_BOOL;
#else
typedef unsigned char MNN_BOOL;
#define true 1
#define false 0
#endif

// Enums from MNN::CV namespace
enum MNN_ImageFormat {
    MNN_RGBA     = 0,
    MNN_RGB      = 1,
    MNN_BGR      = 2,
    MNN_GRAY     = 3,
    MNN_BGRA     = 4,
    MNN_YCrCb    = 5,
    MNN_YUV      = 6,
    MNN_HSV      = 7,
    MNN_XYZ      = 8,
    MNN_BGR555   = 9,
    MNN_BGR565   = 10,
    MNN_YUV_NV21 = 11,
    MNN_YUV_NV12 = 12,
    MNN_YUV_I420 = 13,
    MNN_HSV_FULL = 14,
};

enum MNN_Filter {
    MNN_NEAREST = 0,
    MNN_BILINEAR = 1,
    MNN_BICUBIC = 2
};

enum MNN_Wrap {
    MNN_CLAMP_TO_EDGE = 0,
    MNN_ZERO = 1,
    MNN_REPEAT = 2
};

// Forward declarations
struct MNN_ImageProcess;
struct MNN_ImageProcess_Inside;

// Config struct for ImageProcess
struct MNN_ImageProcess_Config {
    enum MNN_Filter filterType;
    enum MNN_ImageFormat sourceFormat;
    enum MNN_ImageFormat destFormat;
    float mean[4];
    float normal[4];
    enum MNN_Wrap wrap;
};

// Default config values
//#define MNN_IMAGE_PROCESS_CONFIG_DEFAULT {
//constexpr MNN_ImageProcess_Config MNN_IMAGE_PROCESS_CONFIG_DEFAULT = {    
//    MNN_NEAREST,          /* filterType */
//    MNN_RGBA,             /* sourceFormat */
//    MNN_RGBA,             /* destFormat */
//    {0.0f, 0.0f, 0.0f, 0.0f}, /* mean */
//    {1.0f, 1.0f, 1.0f, 1.0f}, /* normal */
//    MNN_CLAMP_TO_EDGE     /* wrap */
//};

#ifdef __cplusplus
extern "C" {
#endif
#include <MNN/HalideRuntime.h>

// ImageProcess methods
MNN_C_API struct MNN_ImageProcess* MNN_ImageProcess_create(const struct MNN_ImageProcess_Config* config, const struct MNN_Tensor* dstTensor);
MNN_C_API struct MNN_ImageProcess* MNN_ImageProcess_create_v2(enum MNN_ImageFormat sourceFormat, enum MNN_ImageFormat destFormat,
                                                   const float* means, int meanCount, const float* normals,
                                                   int normalCount, const struct MNN_Tensor* dstTensor);

MNN_C_API void MNN_ImageProcess_destroy(struct MNN_ImageProcess* imageProcess);

// Matrix operations
// Note: Since Matrix is a separate class, we'll expose limited functionality here
// as the user hasn't asked to convert Matrix.hpp yet
MNN_C_API void MNN_ImageProcess_setMatrix(struct MNN_ImageProcess* imageProcess, const struct MNN_Matrix* matrix);

// Conversion methods
MNN_C_API MNN_ErrorCode MNN_ImageProcess_convert(const struct MNN_ImageProcess* imageProcess, const uint8_t* source, int iw, int ih, int stride, struct MNN_Tensor* dest);
MNN_C_API MNN_ErrorCode MNN_ImageProcess_convert_v2(const struct MNN_ImageProcess* imageProcess, const uint8_t* source, int iw, int ih, int stride,
                                         void* dest, int ow, int oh, int outputBpp, int outputStride, struct halide_type_t type);

// Tensor creation
MNN_C_API struct MNN_Tensor* MNN_ImageProcess_createImageTensor(struct halide_type_t type, int w, int h, int bpp, void* p);

// Utility methods
MNN_C_API void MNN_ImageProcess_setPadding(struct MNN_ImageProcess* imageProcess, uint8_t value);
MNN_C_API void MNN_ImageProcess_setDraw(struct MNN_ImageProcess* imageProcess);
MNN_C_API void MNN_ImageProcess_draw(struct MNN_ImageProcess* imageProcess, uint8_t* img, int w, int h, int c, const int* regions, int num, const uint8_t* color);

#ifdef __cplusplus
}
#endif

#endif /* MNN_ImageProcess_c_h */
