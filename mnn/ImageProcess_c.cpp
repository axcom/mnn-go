// ImageProcess_c.cpp
// MNN
//
// Created by MNN on 2018/09/19.
// Copyright © 2018, Alibaba Group Holding Limited
//

#include <ImageProcess_c.h>
#include <ErrorCode_c.h>
#include "MNN/ImageProcess.hpp"
#include "MNN/Tensor.hpp"
#include "MNN/Matrix.h"

using namespace MNN;
using namespace MNN::CV;

// Helper functions to convert between C and C++ types
static ImageFormat convertToCppImageFormat(enum MNN_ImageFormat format) {
    return static_cast<ImageFormat>(format);
}

static Filter convertToCppFilter(enum MNN_Filter filter) {
    return static_cast<Filter>(filter);
}

static Wrap convertToCppWrap(enum MNN_Wrap wrap) {
    return static_cast<Wrap>(wrap);
}

static ImageProcess::Config convertToCppConfig(const struct MNN_ImageProcess_Config* cConfig) {
    ImageProcess::Config cppConfig;
    cppConfig.filterType = convertToCppFilter(cConfig->filterType);
    cppConfig.sourceFormat = convertToCppImageFormat(cConfig->sourceFormat);
    cppConfig.destFormat = convertToCppImageFormat(cConfig->destFormat);
    for (int i = 0; i < 4; i++) {
        cppConfig.mean[i] = cConfig->mean[i];
        cppConfig.normal[i] = cConfig->normal[i];
    }
    cppConfig.wrap = convertToCppWrap(cConfig->wrap);
    return cppConfig;
}

// ImageProcess methods
struct MNN_ImageProcess* MNN_ImageProcess_create(const struct MNN_ImageProcess_Config* config, const struct MNN_Tensor* dstTensor) {
    const Tensor* cppTensor = nullptr;
    if (dstTensor != nullptr) {
        cppTensor = reinterpret_cast<const Tensor*>(dstTensor);
    }
    ImageProcess::Config cppConfig;
    if (config != nullptr) {
        cppConfig = convertToCppConfig(config);
    }
    ImageProcess* cppProcess = ImageProcess::create(cppConfig, cppTensor);
    return reinterpret_cast<struct MNN_ImageProcess*>(cppProcess);
}

struct MNN_ImageProcess* MNN_ImageProcess_create_v2(enum MNN_ImageFormat sourceFormat, enum MNN_ImageFormat destFormat,
                                                   const float* means, int meanCount, const float* normals,
                                                   int normalCount, const struct MNN_Tensor* dstTensor) {
    const Tensor* cppTensor = nullptr;
    if (dstTensor != nullptr) {
        cppTensor = reinterpret_cast<const Tensor*>(dstTensor);
    }
    ImageProcess* cppProcess = ImageProcess::create(
        convertToCppImageFormat(sourceFormat),
        convertToCppImageFormat(destFormat),
        means, meanCount, normals, normalCount, cppTensor
    );
    return reinterpret_cast<struct MNN_ImageProcess*>(cppProcess);
}

void MNN_ImageProcess_destroy(struct MNN_ImageProcess* imageProcess) {
    if (imageProcess == nullptr) {
        return;
    }
    ImageProcess* cppProcess = reinterpret_cast<ImageProcess*>(imageProcess);
    ImageProcess::destroy(cppProcess);
}

// Matrix operations
void MNN_ImageProcess_setMatrix(struct MNN_ImageProcess* imageProcess, const struct MNN_Matrix* matrix) {
    if (imageProcess == nullptr) {
        return;
    }
    ImageProcess* cppProcess = reinterpret_cast<ImageProcess*>(imageProcess);
    const Matrix* cppMatrix = reinterpret_cast<const Matrix*>(matrix);
    cppProcess->setMatrix(*cppMatrix);
}

// Conversion methods
MNN_ErrorCode MNN_ImageProcess_convert(const struct MNN_ImageProcess* imageProcess, const uint8_t* source, int iw, int ih, int stride, struct MNN_Tensor* dest) {
    //const ImageProcess* cppProcess = reinterpret_cast<const ImageProcess*>(imageProcess);
    ImageProcess* cppProcess = reinterpret_cast<ImageProcess*>(
        const_cast<MNN_ImageProcess*>(imageProcess)
    );
    Tensor* cppDest = reinterpret_cast<Tensor*>(dest);
    ErrorCode cppError = cppProcess->convert(source, iw, ih, stride, cppDest);
    return static_cast<MNN_ErrorCode>(cppError);
}

MNN_ErrorCode MNN_ImageProcess_convert_v2(const struct MNN_ImageProcess* imageProcess, const uint8_t* source, int iw, int ih, int stride,
                                         void* dest, int ow, int oh, int outputBpp, int outputStride, halide_type_t type) {
    //const ImageProcess* cppProcess = reinterpret_cast<const ImageProcess*>(imageProcess);
    ImageProcess* cppProcess = reinterpret_cast<ImageProcess*>(
        const_cast<MNN_ImageProcess*>(imageProcess)
    );
    ErrorCode cppError = cppProcess->convert(source, iw, ih, stride, dest, ow, oh, outputBpp, outputStride, type);
    return static_cast<MNN_ErrorCode>(cppError);
}

// Tensor creation
struct MNN_Tensor* MNN_ImageProcess_createImageTensor(halide_type_t type, int w, int h, int bpp, void* p) {
    Tensor* cppTensor = ImageProcess::createImageTensor(type, w, h, bpp, p);
    return reinterpret_cast<struct MNN_Tensor*>(cppTensor);
}

// Utility methods
void MNN_ImageProcess_setPadding(struct MNN_ImageProcess* imageProcess, uint8_t value) {
    if (imageProcess == nullptr) {
        return;
    }
    ImageProcess* cppProcess = reinterpret_cast<ImageProcess*>(imageProcess);
    cppProcess->setPadding(value);
}

void MNN_ImageProcess_setDraw(struct MNN_ImageProcess* imageProcess) {
    if (imageProcess == nullptr) {
        return;
    }
    ImageProcess* cppProcess = reinterpret_cast<ImageProcess*>(imageProcess);
    cppProcess->setDraw();
}

void MNN_ImageProcess_draw(struct MNN_ImageProcess* imageProcess, uint8_t* img, int w, int h, int c, const int* regions, int num, const uint8_t* color) {
    if (imageProcess == nullptr) {
        return;
    }
    ImageProcess* cppProcess = reinterpret_cast<ImageProcess*>(imageProcess);
    cppProcess->draw(img, w, h, c, regions, num, color);
}
