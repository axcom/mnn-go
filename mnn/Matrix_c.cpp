#include "Matrix_c.h"
#include "MNN/Matrix.h"
#include <MNN/Rect.h>
#include <cstdlib>

using namespace MNN::CV;

// -------------------------- 构造/销毁 --------------------------
MNN_Matrix* MNN_Matrix_Create() {
    auto m = new Matrix();
    return reinterpret_cast<struct MNN_Matrix*>(m);
}

MNN_Matrix* MNN_Matrix_MakeScale(float sx, float sy) {
    auto m = new Matrix(Matrix::MakeScale(sx, sy));
    return reinterpret_cast<struct MNN_Matrix*>(m);
}

MNN_Matrix* MNN_Matrix_MakeScaleUniform(float scale) {
    auto m = new Matrix(Matrix::MakeScale(scale, scale));
    return reinterpret_cast<struct MNN_Matrix*>(m);
}

MNN_Matrix* MNN_Matrix_MakeTrans(float dx, float dy) {
    auto m = new Matrix(Matrix::MakeTrans(dx, dy));
    return reinterpret_cast<struct MNN_Matrix*>(m);
}

MNN_Matrix* MNN_Matrix_MakeAll(float scaleX, float skewX, float transX, float skewY, float scaleY, float transY, float persp0, float persp1, float persp2) {
    auto m = new Matrix(Matrix::MakeAll(scaleX, skewX, transX, skewY, scaleY, transY, persp0, persp1, persp2));
    return reinterpret_cast<struct MNN_Matrix*>(m);
}

MNN_Matrix* MNN_Matrix_MakeRectToRect(const MNN_Rect* src, const MNN_Rect* dst, MNN_Matrix_ScaleToFit stf) {
    Rect c_src{src->left, src->top, src->right, src->bottom};
    Rect c_dst{dst->left, dst->top, dst->right, dst->bottom};
    auto m = new Matrix(Matrix::MakeRectToRect(c_src, c_dst, (Matrix::ScaleToFit)stf));
    return reinterpret_cast<struct MNN_Matrix*>(m);
}

void MNN_Matrix_Destroy(MNN_Matrix* matrix) {
    delete matrix;
}

// -------------------------- 类型判断 --------------------------
MNN_Matrix_TypeMask MNN_Matrix_getType(const MNN_Matrix* matrix) {
    const Matrix* cppMatrix = reinterpret_cast<const Matrix*>(matrix);
    return (MNN_Matrix_TypeMask)cppMatrix->getType();
}

bool MNN_Matrix_isIdentity(const MNN_Matrix* matrix) {
    return reinterpret_cast<const Matrix*>(matrix)->isIdentity();
}

bool MNN_Matrix_isScaleTranslate(const MNN_Matrix* matrix) {
    return reinterpret_cast<const Matrix*>(matrix)->isScaleTranslate();
}

bool MNN_Matrix_isTranslate(const MNN_Matrix* matrix) {
    return reinterpret_cast<const Matrix*>(matrix)->isTranslate();
}

bool MNN_Matrix_rectStaysRect(const MNN_Matrix* matrix) {
    return reinterpret_cast<const Matrix*>(matrix)->rectStaysRect();
}

bool MNN_Matrix_preservesAxisAlignment(const MNN_Matrix* matrix) {
    return reinterpret_cast<const Matrix*>(matrix)->preservesAxisAlignment();
}

// -------------------------- 元素访问/修改 --------------------------
float MNN_Matrix_get(const MNN_Matrix* matrix, int index) {
    return reinterpret_cast<const Matrix*>(matrix)->get(index);
}

void MNN_Matrix_set(MNN_Matrix* matrix, int index, float value) {
    reinterpret_cast<Matrix*>(matrix)->set(index, value);
}

float MNN_Matrix_getScaleX(const MNN_Matrix* matrix) {
    return reinterpret_cast<const Matrix*>(matrix)->getScaleX();
}

float MNN_Matrix_getScaleY(const MNN_Matrix* matrix) {
    return reinterpret_cast<const Matrix*>(matrix)->getScaleY();
}

float MNN_Matrix_getSkewX(const MNN_Matrix* matrix) {
    return reinterpret_cast<const Matrix*>(matrix)->getSkewX();
}

float MNN_Matrix_getSkewY(const MNN_Matrix* matrix) {
    return reinterpret_cast<const Matrix*>(matrix)->getSkewY();
}

float MNN_Matrix_getTranslateX(const MNN_Matrix* matrix) {
    return reinterpret_cast<const Matrix*>(matrix)->getTranslateX();
}

float MNN_Matrix_getTranslateY(const MNN_Matrix* matrix) {
    return reinterpret_cast<const Matrix*>(matrix)->getTranslateY();
}

float MNN_Matrix_getPerspX(const MNN_Matrix* matrix) {
    return reinterpret_cast<const Matrix*>(matrix)->getPerspX();
}

float MNN_Matrix_getPerspY(const MNN_Matrix* matrix) {
    return reinterpret_cast<const Matrix*>(matrix)->getPerspY();
}

void MNN_Matrix_setScaleX(MNN_Matrix* matrix, float v) {
    reinterpret_cast<Matrix*>(matrix)->setScaleX(v);
}

void MNN_Matrix_setScaleY(MNN_Matrix* matrix, float v) {
    reinterpret_cast<Matrix*>(matrix)->setScaleY(v);
}

void MNN_Matrix_setSkewX(MNN_Matrix* matrix, float v) {
    reinterpret_cast<Matrix*>(matrix)->setSkewX(v);
}

void MNN_Matrix_setSkewY(MNN_Matrix* matrix, float v) {
    reinterpret_cast<Matrix*>(matrix)->setSkewY(v);
}

void MNN_Matrix_setTranslateX(MNN_Matrix* matrix, float v) {
    reinterpret_cast<Matrix*>(matrix)->setTranslateX(v);
}

void MNN_Matrix_setTranslateY(MNN_Matrix* matrix, float v) {
    reinterpret_cast<Matrix*>(matrix)->setTranslateY(v);
}

void MNN_Matrix_setPerspX(MNN_Matrix* matrix, float v) {
    reinterpret_cast<Matrix*>(matrix)->setPerspX(v);
}

void MNN_Matrix_setPerspY(MNN_Matrix* matrix, float v) {
    reinterpret_cast<Matrix*>(matrix)->setPerspY(v);
}

void MNN_Matrix_get9(const MNN_Matrix* matrix, float buffer[9]) {
    reinterpret_cast<const Matrix*>(matrix)->get9(buffer);
}

void MNN_Matrix_set9(MNN_Matrix* matrix, const float buffer[9]) {
    reinterpret_cast<Matrix*>(matrix)->set9(buffer);
}

void MNN_Matrix_setAll(MNN_Matrix* matrix, float scaleX, float skewX, float transX, float skewY, float scaleY, float transY, float persp0, float persp1, float persp2) {
    reinterpret_cast<Matrix*>(matrix)->setAll(scaleX, skewX, transX, skewY, scaleY, transY, persp0, persp1, persp2);
}

// -------------------------- 变换设置 --------------------------
void MNN_Matrix_reset(MNN_Matrix* matrix) {
    reinterpret_cast<Matrix*>(matrix)->reset();
}

void MNN_Matrix_setIdentity(MNN_Matrix* matrix) {
    reinterpret_cast<Matrix*>(matrix)->setIdentity();
}

void MNN_Matrix_setTranslate(MNN_Matrix* matrix, float dx, float dy) {
    reinterpret_cast<Matrix*>(matrix)->setTranslate(dx, dy);
}

void MNN_Matrix_setScale(MNN_Matrix* matrix, float sx, float sy) {
    reinterpret_cast<Matrix*>(matrix)->setScale(sx, sy);
}

void MNN_Matrix_setScaleWithPivot(MNN_Matrix* matrix, float sx, float sy, float px, float py) {
    reinterpret_cast<Matrix*>(matrix)->setScale(sx, sy, px, py);
}

void MNN_Matrix_setRotate(MNN_Matrix* matrix, float degrees) {
    reinterpret_cast<Matrix*>(matrix)->setRotate(degrees);
}

void MNN_Matrix_setRotateWithPivot(MNN_Matrix* matrix, float degrees, float px, float py) {
    reinterpret_cast<Matrix*>(matrix)->setRotate(degrees, px, py);
}

void MNN_Matrix_setSinCos(MNN_Matrix* matrix, float sinValue, float cosValue) {
    reinterpret_cast<Matrix*>(matrix)->setSinCos(sinValue, cosValue);
}

void MNN_Matrix_setSinCosWithPivot(MNN_Matrix* matrix, float sinValue, float cosValue, float px, float py) {
    reinterpret_cast<Matrix*>(matrix)->setSinCos(sinValue, cosValue, px, py);
}

void MNN_Matrix_setSkew(MNN_Matrix* matrix, float kx, float ky) {
    reinterpret_cast<Matrix*>(matrix)->setSkew(kx, ky);
}

void MNN_Matrix_setSkewWithPivot(MNN_Matrix* matrix, float kx, float ky, float px, float py) {
    reinterpret_cast<Matrix*>(matrix)->setSkew(kx, ky, px, py);
}

void MNN_Matrix_setScaleTranslate(MNN_Matrix* matrix, float sx, float sy, float tx, float ty) {
    reinterpret_cast<Matrix*>(matrix)->setScaleTranslate(sx, sy, tx, ty);
}

// -------------------------- 矩阵拼接 --------------------------
void MNN_Matrix_setConcat(MNN_Matrix* matrix, const MNN_Matrix* a, const MNN_Matrix* b) {
    reinterpret_cast<Matrix*>(matrix)->setConcat(*reinterpret_cast<const Matrix*>(a), *reinterpret_cast<const Matrix*>(b));
}

void MNN_Matrix_preTranslate(MNN_Matrix* matrix, float dx, float dy) {
    reinterpret_cast<Matrix*>(matrix)->preTranslate(dx, dy);
}

void MNN_Matrix_preScale(MNN_Matrix* matrix, float sx, float sy) {
    reinterpret_cast<Matrix*>(matrix)->preScale(sx, sy);
}

void MNN_Matrix_preScaleWithPivot(MNN_Matrix* matrix, float sx, float sy, float px, float py) {
    reinterpret_cast<Matrix*>(matrix)->preScale(sx, sy, px, py);
}

void MNN_Matrix_preRotate(MNN_Matrix* matrix, float degrees) {
    reinterpret_cast<Matrix*>(matrix)->preRotate(degrees);
}

void MNN_Matrix_preRotateWithPivot(MNN_Matrix* matrix, float degrees, float px, float py) {
    reinterpret_cast<Matrix*>(matrix)->preRotate(degrees, px, py);
}

void MNN_Matrix_preSkew(MNN_Matrix* matrix, float kx, float ky) {
    reinterpret_cast<Matrix*>(matrix)->preSkew(kx, ky);
}

void MNN_Matrix_preSkewWithPivot(MNN_Matrix* matrix, float kx, float ky, float px, float py) {
    reinterpret_cast<Matrix*>(matrix)->preSkew(kx, ky, px, py);
}

void MNN_Matrix_preConcat(MNN_Matrix* matrix, const MNN_Matrix* other) {
    reinterpret_cast<Matrix*>(matrix)->preConcat(*reinterpret_cast<const Matrix*>(other));
}

void MNN_Matrix_postTranslate(MNN_Matrix* matrix, float dx, float dy) {
    reinterpret_cast<Matrix*>(matrix)->postTranslate(dx, dy);
}

void MNN_Matrix_postScale(MNN_Matrix* matrix, float sx, float sy) {
    reinterpret_cast<Matrix*>(matrix)->postScale(sx, sy);
}

void MNN_Matrix_postScaleWithPivot(MNN_Matrix* matrix, float sx, float sy, float px, float py) {
    reinterpret_cast<Matrix*>(matrix)->postScale(sx, sy, px, py);
}

bool MNN_Matrix_postIDiv(MNN_Matrix* matrix, int divx, int divy) {
    return reinterpret_cast<Matrix*>(matrix)->postIDiv(divx, divy);
}

void MNN_Matrix_postRotate(MNN_Matrix* matrix, float degrees) {
    reinterpret_cast<Matrix*>(matrix)->postRotate(degrees);
}

void MNN_Matrix_postRotateWithPivot(MNN_Matrix* matrix, float degrees, float px, float py) {
    reinterpret_cast<Matrix*>(matrix)->postRotate(degrees, px, py);
}

void MNN_Matrix_postSkew(MNN_Matrix* matrix, float kx, float ky) {
    reinterpret_cast<Matrix*>(matrix)->postSkew(kx, ky);
}

void MNN_Matrix_postSkewWithPivot(MNN_Matrix* matrix, float kx, float ky, float px, float py) {
    reinterpret_cast<Matrix*>(matrix)->postSkew(kx, ky, px, py);
}

void MNN_Matrix_postConcat(MNN_Matrix* matrix, const MNN_Matrix* other) {
    reinterpret_cast<Matrix*>(matrix)->postConcat(*reinterpret_cast<const Matrix*>(other));
}

// -------------------------- 变换应用 --------------------------
bool MNN_Matrix_setRectToRect(MNN_Matrix* matrix, const MNN_Rect* src, const MNN_Rect* dst, MNN_Matrix_ScaleToFit stf) {
    Rect c_src{src->left, src->top, src->right, src->bottom};
    Rect c_dst{dst->left, dst->top, dst->right, dst->bottom};
    return reinterpret_cast<Matrix*>(matrix)->setRectToRect(c_src, c_dst, (Matrix::ScaleToFit)stf);
}

bool MNN_Matrix_setPolyToPoly(MNN_Matrix* matrix, const MNN_Point src[], const MNN_Point dst[], int count) {
    Point* c_src = new Point[count];
    Point* c_dst = new Point[count];
    for (int i = 0; i < count; i++) {
        c_src[i].fX = src[i].x;
        c_src[i].fY = src[i].y;
        c_dst[i].fX = dst[i].x;
        c_dst[i].fY = dst[i].y;
    }
    bool ret = reinterpret_cast<Matrix*>(matrix)->setPolyToPoly(c_src, c_dst, count);
    delete[] c_src;
    delete[] c_dst;
    return ret;
}

bool MNN_Matrix_invert(const MNN_Matrix* matrix, MNN_Matrix* inverse) {
    return reinterpret_cast<const Matrix*>(matrix)->invert(reinterpret_cast<Matrix*>(inverse));
}

void MNN_Matrix_SetAffineIdentity(float affine[6]) {
    Matrix::SetAffineIdentity(affine);
}

bool MNN_Matrix_asAffine(const MNN_Matrix* matrix, float affine[6]) {
    return reinterpret_cast<const Matrix*>(matrix)->asAffine(affine);
}

void MNN_Matrix_setAffine(MNN_Matrix* matrix, const float affine[6]) {
    reinterpret_cast<Matrix*>(matrix)->setAffine(affine);
}

/*void MNN_Matrix_mapPoints(const MNN_Matrix* matrix, MNN_Point dst[], const MNN_Point src[], int count) {
    Point* c_src = new Point[count];
    Point* c_dst = new Point[count];
    for (int i = 0; i < count; i++) {
        c_src[i].fX = src[i].x;
        c_src[i].fY = src[i].y;
    }
    reinterpret_cast<const Matrix*>(matrix)->mapPoints(c_dst, c_src, count);
    for (int i = 0; i < count; i++) {
        dst[i].x = c_dst[i].fX;
        dst[i].y = c_dst[i].fY;
    }
    delete[] c_src;
    delete[] c_dst;
}

void MNN_Matrix_mapPointsInPlace(const MNN_Matrix* matrix, MNN_Point pts[], int count) {
    Point* c_pts = new Point[count];
    for (int i = 0; i < count; i++) {
        c_pts[i].fX = pts[i].x;
        c_pts[i].fY = pts[i].y;
    }
    reinterpret_cast<const Matrix*>(matrix)->mapPoints(c_pts, count);
    for (int i = 0; i < count; i++) {
        pts[i].x = c_pts[i].fX;
        pts[i].y = c_pts[i].fY;
    }
    delete[] c_pts;
}

MNN_Point MNN_Matrix_mapXY(const MNN_Matrix* matrix, float x, float y) {
    Point p = reinterpret_cast<const Matrix*>(matrix)->mapXY(x, y);
    return {p.fX, p.fY};
}*/

//MNN::CV::Matrix 类中的 gMapXYProcs 是一个【私有静态成员变量】，MNN 官方编译 MNN.dll 时，这个私有静态变量没有被导出到 DLL 的符号表中 **，且是 private 私有访问权限，外部无法访问；同时 mapXY() 函数的实现强依赖这个私有静态数组，导致链接器找不到它的实现，直接报 LNK2019 错误
//mapXY 是 MNN.dll 的导出公有函数，理论上外部调用它时，只需要调用这个导出函数即可，完全不应该感知到、也不应该依赖 Matrix 类的私有成员 gMapXYProcs —— 你的这个认知是绝对正确的，这也是 C++ 编译 DLL 的标准工程规范！
//但使用的 MNN.dll，被编译为「静态链接的类库（静态导出）」，而非「动态链接的类库（动态导出）」，导致 Matrix 的成员函数的实现体代码，没有被编译进 MNN.dll 内部 **，而是「延迟到外部调用时，在你的工程里二次编译链接」。**这是所有问题的总根源！
//真相：MNN.dll 是「头文件内联实现 + 符号导出不完整」的「伪 DLL」！MNN 的开发者在实现Matrix类时，为了追求效率，把绝大多数成员函数的实现体，直接写在了 Matrix.h 头文件的类内（即内联函数 / 类内实现），而非写在.cpp文件中编译进 DLL。

bool MNN_Matrix_mapRect(const MNN_Matrix* matrix, MNN_Rect* dst, const MNN_Rect* src) {
    Rect c_src{src->left, src->top, src->right, src->bottom};
    Rect c_dst;
    bool ret = reinterpret_cast<const Matrix*>(matrix)->mapRect(&c_dst, c_src);
    dst->left = c_dst.left();
    dst->top = c_dst.top();
    dst->right = c_dst.right();
    dst->bottom = c_dst.bottom();
    return ret;
}

bool MNN_Matrix_mapRectInPlace(const MNN_Matrix* matrix, MNN_Rect* rect) {
    Rect c_rect{rect->left, rect->top, rect->right, rect->bottom};
    bool ret = reinterpret_cast<const Matrix*>(matrix)->mapRect(&c_rect);
    rect->left = c_rect.left();
    rect->top = c_rect.top();
    rect->right = c_rect.right();
    rect->bottom = c_rect.bottom();
    return ret;
}

void MNN_Matrix_mapRectScaleTranslate(const MNN_Matrix* matrix, MNN_Rect* dst, const MNN_Rect* src) {
    Rect c_src{src->left, src->top, src->right, src->bottom};
    Rect c_dst;
    reinterpret_cast<const Matrix*>(matrix)->mapRectScaleTranslate(&c_dst, c_src);
    dst->left = c_dst.left();
    dst->top = c_dst.top();
    dst->right = c_dst.right();
    dst->bottom = c_dst.bottom();
}

// -------------------------- 比较/调试 --------------------------
bool MNN_Matrix_cheapEqualTo(const MNN_Matrix* a, const MNN_Matrix* b) {
    return reinterpret_cast<const Matrix*>(a)->cheapEqualTo(*reinterpret_cast<const Matrix*>(b));
}

bool MNN_Matrix_equals(const MNN_Matrix* a, const MNN_Matrix* b) {
    return reinterpret_cast<const Matrix*>(a) == reinterpret_cast<const Matrix*>(b);
}

bool MNN_Matrix_notEquals(const MNN_Matrix* a, const MNN_Matrix* b) {
    return reinterpret_cast<const Matrix*>(a) != reinterpret_cast<const Matrix*>(b);
}

/*void MNN_Matrix_dump(const MNN_Matrix* matrix) {
    reinterpret_cast<const Matrix*>(matrix)->dump();
}*/

// -------------------------- 缩放因子 --------------------------
/*float MNN_Matrix_getMinScale(const MNN_Matrix* matrix) {
    return reinterpret_cast<const Matrix*>(matrix)->getMinScale();
}

float MNN_Matrix_getMaxScale(const MNN_Matrix* matrix) {
    return reinterpret_cast<const Matrix*>(matrix)->getMaxScale();
}

bool MNN_Matrix_getMinMaxScales(const MNN_Matrix* matrix, float scaleFactors[2]) {
    return reinterpret_cast<const Matrix*>(matrix)->getMinMaxScales(scaleFactors);
}*/

//error LNK2019: 无法解析的外部符号 "public: float __cdecl MNN::CV::Matrix::getMinScale(void)const " (?getMinScale@Matrix@CV@MNN@@QEBAMXZ)，函数 MNN_Matrix_getMinScale 中引用了该符号
//MNN.dll 编译时，这 3 个函数被编译器优化为「内部符号」，没有导出到 DLL 的符号表中
//因为是「只读、无修改、纯计算型」的简单函数，MNN 官方编译时，编译器做了 inline/内部符号 优化 → 没有被导出到 MNN.dll 的符号表中

// -------------------------- 工具函数 --------------------------
/*const MNN_Matrix* MNN_Matrix_Identity() {
    static MNN_Matrix identity{Matrix::I()};
    return &identity;
}*/

/*const MNN_Matrix* MNN_Matrix_InvalidMatrix() {
    static MNN_Matrix invalid{Matrix::InvalidMatrix()};
    return &invalid;
}*/

MNN_Matrix* MNN_Matrix_Concat(const MNN_Matrix* a, const MNN_Matrix* b) {
    auto m = new Matrix();
    reinterpret_cast<Matrix*>(m)->Concat(*reinterpret_cast<const Matrix*>(a), *reinterpret_cast<const Matrix*>(b));
    return reinterpret_cast<MNN_Matrix*>(m);
}

void MNN_Matrix_dirtyMatrixTypeCache(MNN_Matrix* matrix) {
    reinterpret_cast<Matrix*>(matrix)->dirtyMatrixTypeCache();
}