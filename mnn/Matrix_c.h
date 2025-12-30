#ifndef MNN_MATRIX_C_H
#define MNN_MATRIX_C_H

// 导出宏：Windows下导出，其他平台兼容
#ifdef MNN_C_EXPORTS
    // 仅当编译 libmnn.dll 时定义 MNN_C_EXPORTS，此时用 dllexport
    #define MNN_C_API __declspec(dllexport)
#else
    #define MNN_C_API
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// 保留原C++的常量名
typedef enum {
    kIdentity_Mask = 0,
    kTranslate_Mask = 0x01,
    kScale_Mask = 0x02,
    kAffine_Mask = 0x04,
    kPerspective_Mask = 0x08
} MNN_Matrix_TypeMask;

typedef enum {
    kFill_ScaleToFit,
    kStart_ScaleToFit,
    kCenter_ScaleToFit,
    kEnd_ScaleToFit
} MNN_Matrix_ScaleToFit;

// 矩阵元素索引常量（保留原C++名）
enum {
    kMScaleX = 0,
    kMSkewX = 1,
    kMTransX = 2,
    kMSkewY = 3,
    kMScaleY = 4,
    kMTransY = 5,
    kMPersp0 = 6,
    kMPersp1 = 7,
    kMPersp2 = 8
};

enum {
    kAScaleX = 0,
    kASkewY = 1,
    kASkewX = 2,
    kAScaleY = 3,
    kATransX = 4,
    kATransY = 5
};

// 基础数据结构（对应MNN::CV::Point/Rect）
typedef struct MNN_Point {
    float x;
    float y;
} MNN_Point;

typedef struct MNN_Rect {
    float left;
    float top;
    float right;
    float bottom;
} MNN_Rect;

// 不透明结构体，隐藏C++ Matrix实现
//struct MNN_Matrix;
typedef struct MNN_Matrix //MNN_Matrix;
{
    float fMat[9];          // 矩阵元素（行优先）
    uint32_t fTypeMask;     // 类型掩码（缓存）
} MNN_Matrix;

#ifdef __cplusplus
extern "C" {
#endif

// -------------------------- 构造/销毁 --------------------------
MNN_C_API MNN_Matrix* MNN_Matrix_Create();
MNN_C_API MNN_Matrix* MNN_Matrix_MakeScale(float sx, float sy);
MNN_C_API MNN_Matrix* MNN_Matrix_MakeScaleUniform(float scale);
MNN_C_API MNN_Matrix* MNN_Matrix_MakeTrans(float dx, float dy);
MNN_C_API MNN_Matrix* MNN_Matrix_MakeAll(float scaleX, float skewX, float transX, float skewY, float scaleY, float transY, float persp0, float persp1, float persp2);
MNN_C_API MNN_Matrix* MNN_Matrix_MakeRectToRect(const MNN_Rect* src, const MNN_Rect* dst, MNN_Matrix_ScaleToFit stf);
MNN_C_API void MNN_Matrix_Destroy(MNN_Matrix* matrix);

// -------------------------- 类型判断 --------------------------
MNN_C_API MNN_Matrix_TypeMask MNN_Matrix_getType(const MNN_Matrix* matrix);
MNN_C_API bool MNN_Matrix_isIdentity(const MNN_Matrix* matrix);
MNN_C_API bool MNN_Matrix_isScaleTranslate(const MNN_Matrix* matrix);
MNN_C_API bool MNN_Matrix_isTranslate(const MNN_Matrix* matrix);
MNN_C_API bool MNN_Matrix_rectStaysRect(const MNN_Matrix* matrix);
MNN_C_API bool MNN_Matrix_preservesAxisAlignment(const MNN_Matrix* matrix);

// -------------------------- 元素访问/修改 --------------------------
MNN_C_API float MNN_Matrix_get(const MNN_Matrix* matrix, int index);
MNN_C_API void MNN_Matrix_set(MNN_Matrix* matrix, int index, float value);
MNN_C_API float MNN_Matrix_getScaleX(const MNN_Matrix* matrix);
MNN_C_API float MNN_Matrix_getScaleY(const MNN_Matrix* matrix);
MNN_C_API float MNN_Matrix_getSkewX(const MNN_Matrix* matrix);
MNN_C_API float MNN_Matrix_getSkewY(const MNN_Matrix* matrix);
MNN_C_API float MNN_Matrix_getTranslateX(const MNN_Matrix* matrix);
MNN_C_API float MNN_Matrix_getTranslateY(const MNN_Matrix* matrix);
MNN_C_API float MNN_Matrix_getPerspX(const MNN_Matrix* matrix);
MNN_C_API float MNN_Matrix_getPerspY(const MNN_Matrix* matrix);

MNN_C_API void MNN_Matrix_setScaleX(MNN_Matrix* matrix, float v);
MNN_C_API void MNN_Matrix_setScaleY(MNN_Matrix* matrix, float v);
MNN_C_API void MNN_Matrix_setSkewX(MNN_Matrix* matrix, float v);
MNN_C_API void MNN_Matrix_setSkewY(MNN_Matrix* matrix, float v);
MNN_C_API void MNN_Matrix_setTranslateX(MNN_Matrix* matrix, float v);
MNN_C_API void MNN_Matrix_setTranslateY(MNN_Matrix* matrix, float v);
MNN_C_API void MNN_Matrix_setPerspX(MNN_Matrix* matrix, float v);
MNN_C_API void MNN_Matrix_setPerspY(MNN_Matrix* matrix, float v);

MNN_C_API void MNN_Matrix_get9(const MNN_Matrix* matrix, float buffer[9]);
MNN_C_API void MNN_Matrix_set9(MNN_Matrix* matrix, const float buffer[9]);
MNN_C_API void MNN_Matrix_setAll(MNN_Matrix* matrix, float scaleX, float skewX, float transX, float skewY, float scaleY, float transY, float persp0, float persp1, float persp2);

// -------------------------- 变换设置 --------------------------
MNN_C_API void MNN_Matrix_reset(MNN_Matrix* matrix);
MNN_C_API void MNN_Matrix_setIdentity(MNN_Matrix* matrix);
MNN_C_API void MNN_Matrix_setTranslate(MNN_Matrix* matrix, float dx, float dy);
MNN_C_API void MNN_Matrix_setScale(MNN_Matrix* matrix, float sx, float sy);
MNN_C_API void MNN_Matrix_setScaleWithPivot(MNN_Matrix* matrix, float sx, float sy, float px, float py);
MNN_C_API void MNN_Matrix_setRotate(MNN_Matrix* matrix, float degrees);
MNN_C_API void MNN_Matrix_setRotateWithPivot(MNN_Matrix* matrix, float degrees, float px, float py);
MNN_C_API void MNN_Matrix_setSinCos(MNN_Matrix* matrix, float sinValue, float cosValue);
MNN_C_API void MNN_Matrix_setSinCosWithPivot(MNN_Matrix* matrix, float sinValue, float cosValue, float px, float py);
MNN_C_API void MNN_Matrix_setSkew(MNN_Matrix* matrix, float kx, float ky);
MNN_C_API void MNN_Matrix_setSkewWithPivot(MNN_Matrix* matrix, float kx, float ky, float px, float py);
MNN_C_API void MNN_Matrix_setScaleTranslate(MNN_Matrix* matrix, float sx, float sy, float tx, float ty);

// -------------------------- 矩阵拼接 --------------------------
MNN_C_API void MNN_Matrix_setConcat(MNN_Matrix* matrix, const MNN_Matrix* a, const MNN_Matrix* b);
MNN_C_API void MNN_Matrix_preTranslate(MNN_Matrix* matrix, float dx, float dy);
MNN_C_API void MNN_Matrix_preScale(MNN_Matrix* matrix, float sx, float sy);
MNN_C_API void MNN_Matrix_preScaleWithPivot(MNN_Matrix* matrix, float sx, float sy, float px, float py);
MNN_C_API void MNN_Matrix_preRotate(MNN_Matrix* matrix, float degrees);
MNN_C_API void MNN_Matrix_preRotateWithPivot(MNN_Matrix* matrix, float degrees, float px, float py);
MNN_C_API void MNN_Matrix_preSkew(MNN_Matrix* matrix, float kx, float ky);
MNN_C_API void MNN_Matrix_preSkewWithPivot(MNN_Matrix* matrix, float kx, float ky, float px, float py);
MNN_C_API void MNN_Matrix_preConcat(MNN_Matrix* matrix, const MNN_Matrix* other);

MNN_C_API void MNN_Matrix_postTranslate(MNN_Matrix* matrix, float dx, float dy);
MNN_C_API void MNN_Matrix_postScale(MNN_Matrix* matrix, float sx, float sy);
MNN_C_API void MNN_Matrix_postScaleWithPivot(MNN_Matrix* matrix, float sx, float sy, float px, float py);
MNN_C_API bool MNN_Matrix_postIDiv(MNN_Matrix* matrix, int divx, int divy);
MNN_C_API void MNN_Matrix_postRotate(MNN_Matrix* matrix, float degrees);
MNN_C_API void MNN_Matrix_postRotateWithPivot(MNN_Matrix* matrix, float degrees, float px, float py);
MNN_C_API void MNN_Matrix_postSkew(MNN_Matrix* matrix, float kx, float ky);
MNN_C_API void MNN_Matrix_postSkewWithPivot(MNN_Matrix* matrix, float kx, float ky, float px, float py);
MNN_C_API void MNN_Matrix_postConcat(MNN_Matrix* matrix, const MNN_Matrix* other);

// -------------------------- 变换应用 --------------------------
MNN_C_API bool MNN_Matrix_setRectToRect(MNN_Matrix* matrix, const MNN_Rect* src, const MNN_Rect* dst, MNN_Matrix_ScaleToFit stf);
MNN_C_API bool MNN_Matrix_setPolyToPoly(MNN_Matrix* matrix, const MNN_Point src[], const MNN_Point dst[], int count);
MNN_C_API bool MNN_Matrix_invert(const MNN_Matrix* matrix, MNN_Matrix* inverse);
MNN_C_API void MNN_Matrix_SetAffineIdentity(float affine[6]);
MNN_C_API bool MNN_Matrix_asAffine(const MNN_Matrix* matrix, float affine[6]);
MNN_C_API void MNN_Matrix_setAffine(MNN_Matrix* matrix, const float affine[6]);

//MNN_C_API void MNN_Matrix_mapPoints(const MNN_Matrix* matrix, MNN_Point dst[], const MNN_Point src[], int count);
//MNN_C_API void MNN_Matrix_mapPointsInPlace(const MNN_Matrix* matrix, MNN_Point pts[], int count);
//MNN_C_API struct MNN_Point MNN_Matrix_mapXY(const MNN_Matrix* matrix, float x, float y);
MNN_C_API bool MNN_Matrix_mapRect(const MNN_Matrix* matrix, MNN_Rect* dst, const MNN_Rect* src);
MNN_C_API bool MNN_Matrix_mapRectInPlace(const MNN_Matrix* matrix, MNN_Rect* rect);
MNN_C_API void MNN_Matrix_mapRectScaleTranslate(const MNN_Matrix* matrix, MNN_Rect* dst, const MNN_Rect* src);

// -------------------------- 比较/调试 --------------------------
MNN_C_API bool MNN_Matrix_cheapEqualTo(const MNN_Matrix* a, const MNN_Matrix* b);
MNN_C_API bool MNN_Matrix_equals(const MNN_Matrix* a, const MNN_Matrix* b);
MNN_C_API bool MNN_Matrix_notEquals(const MNN_Matrix* a, const MNN_Matrix* b);
//MNN_C_API void MNN_Matrix_dump(const MNN_Matrix* matrix);

// -------------------------- 缩放因子 --------------------------
//MNN_C_API float MNN_Matrix_getMinScale(const MNN_Matrix* matrix);
//MNN_C_API float MNN_Matrix_getMaxScale(const MNN_Matrix* matrix);
//MNN_C_API bool MNN_Matrix_getMinMaxScales(const MNN_Matrix* matrix, float scaleFactors[2]);

// -------------------------- 工具函数 --------------------------
//MNN_C_API const MNN_Matrix* MNN_Matrix_GetIdentity();
//MNN_C_API const MNN_Matrix* MNN_Matrix_GetInvalidMatrix();
MNN_C_API MNN_Matrix* MNN_Matrix_Concat(const MNN_Matrix* a, const MNN_Matrix* b);
MNN_C_API void MNN_Matrix_dirtyMatrixTypeCache(MNN_Matrix* matrix);

#ifdef __cplusplus
}
#endif

#endif // MNN_MATRIX_C_H