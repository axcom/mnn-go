package mnn

/*
#include "Matrix_c.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"math"
	"unsafe"
)

// -------------------------- 保留原C++常量名 --------------------------

// MNN_Matrix_TypeMask 矩阵变换类型掩码（原C++ Matrix::TypeMask）
type MNN_Matrix_TypeMask C.MNN_Matrix_TypeMask

// 保留原C++常量名
const (
	kIdentity_Mask    MNN_Matrix_TypeMask = C.kIdentity_Mask    //单位矩阵
	kTranslate_Mask   MNN_Matrix_TypeMask = C.kTranslate_Mask   //转换矩阵
	kScale_Mask       MNN_Matrix_TypeMask = C.kScale_Mask       //缩放矩阵
	kAffine_Mask      MNN_Matrix_TypeMask = C.kAffine_Mask      //倾斜或旋转矩阵
	kPerspective_Mask MNN_Matrix_TypeMask = C.kPerspective_Mask //透视矩阵
)

// MNN_Matrix_ScaleToFit 缩放适配模式（原C++ Matrix::ScaleToFit）
type MNN_Matrix_ScaleToFit C.MNN_Matrix_ScaleToFit

// 保留原C++常量名
const (
	kFill_ScaleToFit   MNN_Matrix_ScaleToFit = C.kFill_ScaleToFit   //缩放x和y来填充目标矩形
	kStart_ScaleToFit  MNN_Matrix_ScaleToFit = C.kStart_ScaleToFit  //在左和上缩放和对齐
	kCenter_ScaleToFit MNN_Matrix_ScaleToFit = C.kCenter_ScaleToFit //中心缩放和对齐
	kEnd_ScaleToFit    MNN_Matrix_ScaleToFit = C.kEnd_ScaleToFit    //在右边和底部缩放和对齐
)

// 矩阵元素索引常量（保留原C++名）
const (
	kMScaleX = C.kMScaleX
	kMSkewX  = C.kMSkewX
	kMTransX = C.kMTransX
	kMSkewY  = C.kMSkewY
	kMScaleY = C.kMScaleY
	kMTransY = C.kMTransY
	kMPersp0 = C.kMPersp0
	kMPersp1 = C.kMPersp1
	kMPersp2 = C.kMPersp2
)

// 仿射矩阵索引常量（保留原C++名）
const (
	kAScaleX = C.kAScaleX
	kASkewY  = C.kASkewY
	kASkewX  = C.kASkewX
	kAScaleY = C.kAScaleY
	kATransX = C.kATransX
	kATransY = C.kATransY
)

// -------------------------- 数据结构映射 --------------------------

// MNN_Matrix MNN矩阵的Go封装
type Matrix struct {
	fMat      [9]float32 // 矩阵元素 [scaleX, skewX, transX, skewY, scaleY, transY, persp0, persp1, persp2]
	fTypeMask uint32     // 类型掩码缓存
}

// C++ -> Go 转换
func FromCMatrix(cMat *C.MNN_Matrix) *Matrix {
	return (*Matrix)(unsafe.Pointer(cMat))
	/*return &Matrix{
		fMat: [9]float32{
			float32(cMat.fMat[0]), // 显式拷贝数据
			float32(cMat.fMat[1]), // ... 其他8个元素
			float32(cMat.fMat[2]),
			float32(cMat.fMat[3]),
			float32(cMat.fMat[4]),
			float32(cMat.fMat[5]),
			float32(cMat.fMat[6]),
			float32(cMat.fMat[7]),
			float32(cMat.fMat[8]),
		},
		fTypeMask: uint32(cMat.fTypeMask),
	}*/
}

// Go -> C++ 转换: 在确保内存布局一致，通过unsafe转换
func (m *Matrix) UnsafeC() *C.MNN_Matrix {
	return (*C.MNN_Matrix)(unsafe.Pointer(m))
}

// -------------------------- 构造/销毁 --------------------------

// NewMatrix 创建空矩阵（单位矩阵）
func NewMatrix() *Matrix {
	cMat := C.MNN_Matrix_Create()
	if cMat == nil {
		panic("failed to create MNN Matrix")
	}
	return FromCMatrix(cMat)
}

// MakeScale 创建缩放矩阵
func MatrixMakeScale(sx, sy float32) *Matrix {
	cMat := C.MNN_Matrix_MakeScale(C.float(sx), C.float(sy))
	if cMat == nil {
		panic("failed to create scale matrix")
	}
	return FromCMatrix(cMat)
}

// MakeScaleUniform 创建统一缩放矩阵
func MatrixMakeScaleUniform(scale float32) *Matrix {
	cMat := C.MNN_Matrix_MakeScaleUniform(C.float(scale))
	if cMat == nil {
		panic("failed to create uniform scale matrix")
	}
	return FromCMatrix(cMat)
}

// MakeTrans 创建平移矩阵
func MatrixMakeTrans(dx, dy float32) *Matrix {
	cMat := C.MNN_Matrix_MakeTrans(C.float(dx), C.float(dy))
	if cMat == nil {
		panic("failed to create translate matrix")
	}
	return FromCMatrix(cMat)
}

// MakeAll 从9个元素创建矩阵
func MatrixMakeAll(scaleX, skewX, transX, skewY, scaleY, transY, persp0, persp1, persp2 float32) *Matrix {
	cMat := C.MNN_Matrix_MakeAll(
		C.float(scaleX), C.float(skewX), C.float(transX),
		C.float(skewY), C.float(scaleY), C.float(transY),
		C.float(persp0), C.float(persp1), C.float(persp2),
	)
	if cMat == nil {
		panic("failed to create matrix from all elements")
	}
	return FromCMatrix(cMat)
}

// MakeRectToRect 创建矩形映射矩阵
func MatrixMakeRectToRect(src, dst Rect, stf MNN_Matrix_ScaleToFit) *Matrix {
	cMat := C.MNN_Matrix_MakeRectToRect(src.UnsafeC(), dst.UnsafeC(), C.MNN_Matrix_ScaleToFit(stf))
	if cMat == nil {
		panic("failed to create rect-to-rect matrix")
	}
	return FromCMatrix(cMat)
}

// Destroy 销毁矩阵
func (m *Matrix) Close() {
	C.MNN_Matrix_Destroy(m.UnsafeC())
}

// -------------------------- 类型判断 --------------------------

// GetType 获取矩阵类型掩码
func (m *Matrix) GetType() MNN_Matrix_TypeMask {
	return MNN_Matrix_TypeMask(C.MNN_Matrix_getType(m.UnsafeC()))
}

// IsIdentity 是否为单位矩阵
func (m *Matrix) IsIdentity() bool {
	return bool(C.MNN_Matrix_isIdentity(m.UnsafeC()))
}

// IsScaleTranslate 是否仅包含缩放和平移
func (m *Matrix) IsScaleTranslate() bool {
	return bool(C.MNN_Matrix_isScaleTranslate(m.UnsafeC()))
}

// IsTranslate 是否仅包含平移
func (m *Matrix) IsTranslate() bool {
	return bool(C.MNN_Matrix_isTranslate(m.UnsafeC()))
}

// RectStaysRect 是否保持矩形
func (m *Matrix) RectStaysRect() bool {
	return bool(C.MNN_Matrix_rectStaysRect(m.UnsafeC()))
}

// PreservesAxisAlignment 是否保持坐标轴对齐
func (m *Matrix) PreservesAxisAlignment() bool {
	return bool(C.MNN_Matrix_preservesAxisAlignment(m.UnsafeC()))
}

// -------------------------- 元素访问/修改 --------------------------

// Get 获取矩阵指定索引的元素
func (m *Matrix) Get(index int) float32 {
	return float32(C.MNN_Matrix_get(m.UnsafeC(), C.int(index)))
}

// Set 设置矩阵指定索引的元素
func (m *Matrix) Set(index int, value float32) {
	C.MNN_Matrix_set(m.UnsafeC(), C.int(index), C.float(value))
}

// GetScaleX 获取X轴缩放
func (m *Matrix) GetScaleX() float32 {
	return float32(C.MNN_Matrix_getScaleX(m.UnsafeC()))
}

// SetScaleX 设置X轴缩放
func (m *Matrix) SetScaleX(v float32) {
	C.MNN_Matrix_setScaleX(m.UnsafeC(), C.float(v))
}

// GetScaleY 获取Y轴缩放
func (m *Matrix) GetScaleY() float32 {
	return float32(C.MNN_Matrix_getScaleY(m.UnsafeC()))
}

// SetScaleY 设置Y轴缩放
func (m *Matrix) SetScaleY(v float32) {
	C.MNN_Matrix_setScaleY(m.UnsafeC(), C.float(v))
}

// GetSkewX 获取X轴斜切
func (m *Matrix) GetSkewX() float32 {
	return float32(C.MNN_Matrix_getSkewX(m.UnsafeC()))
}

// SetSkewX 设置X轴斜切
func (m *Matrix) SetSkewX(v float32) {
	C.MNN_Matrix_setSkewX(m.UnsafeC(), C.float(v))
}

// GetSkewY 获取Y轴斜切
func (m *Matrix) GetSkewY() float32 {
	return float32(C.MNN_Matrix_getSkewY(m.UnsafeC()))
}

// SetSkewY 设置Y轴斜切
func (m *Matrix) SetSkewY(v float32) {
	C.MNN_Matrix_setSkewY(m.UnsafeC(), C.float(v))
}

// GetTranslateX 获取X轴平移
func (m *Matrix) GetTranslateX() float32 {
	return float32(C.MNN_Matrix_getTranslateX(m.UnsafeC()))
}

// SetTranslateX 设置X轴平移
func (m *Matrix) SetTranslateX(v float32) {
	C.MNN_Matrix_setTranslateX(m.UnsafeC(), C.float(v))
}

// GetTranslateY 获取Y轴平移
func (m *Matrix) GetTranslateY() float32 {
	return float32(C.MNN_Matrix_getTranslateY(m.UnsafeC()))
}

// SetTranslateY 设置Y轴平移
func (m *Matrix) SetTranslateY(v float32) {
	C.MNN_Matrix_setTranslateY(m.UnsafeC(), C.float(v))
}

// GetPerspX 获取X轴透视
func (m *Matrix) GetPerspX() float32 {
	return float32(C.MNN_Matrix_getPerspX(m.UnsafeC()))
}

// SetPerspX 设置X轴透视
func (m *Matrix) SetPerspX(v float32) {
	C.MNN_Matrix_setPerspX(m.UnsafeC(), C.float(v))
}

// GetPerspY 获取Y轴透视
func (m *Matrix) GetPerspY() float32 {
	return float32(C.MNN_Matrix_getPerspY(m.UnsafeC()))
}

// SetPerspY 设置Y轴透视
func (m *Matrix) SetPerspY(v float32) {
	C.MNN_Matrix_setPerspY(m.UnsafeC(), C.float(v))
}

// Get9 获取矩阵的9个元素
func (m *Matrix) Get9() [9]float32 {
	var cBuf [9]C.float
	C.MNN_Matrix_get9(m.UnsafeC(), &cBuf[0])
	var elems [9]float32
	for i := 0; i < 9; i++ {
		elems[i] = float32(cBuf[i])
	}
	return elems
}

// Set9 设置矩阵的9个元素
func (m *Matrix) Set9(elems [9]float32) {
	var cBuf [9]C.float
	for i := 0; i < 9; i++ {
		cBuf[i] = C.float(elems[i])
	}
	C.MNN_Matrix_set9(m.UnsafeC(), &cBuf[0])
}

// SetAll 设置矩阵所有元素
func (m *Matrix) SetAll(scaleX, skewX, transX, skewY, scaleY, transY, persp0, persp1, persp2 float32) {
	C.MNN_Matrix_setAll(
		m.UnsafeC(),
		C.float(scaleX), C.float(skewX), C.float(transX),
		C.float(skewY), C.float(scaleY), C.float(transY),
		C.float(persp0), C.float(persp1), C.float(persp2),
	)
}

// -------------------------- 变换设置 --------------------------

// Reset 重置为单位矩阵
func (m *Matrix) Reset() {
	C.MNN_Matrix_reset(m.UnsafeC())
}

// SetIdentity 设置为单位矩阵
func (m *Matrix) SetIdentity() {
	C.MNN_Matrix_setIdentity(m.UnsafeC())
}

// SetTranslate 设置平移
func (m *Matrix) SetTranslate(dx, dy float32) {
	C.MNN_Matrix_setTranslate(m.UnsafeC(), C.float(dx), C.float(dy))
}

// SetScale 设置缩放（绕原点）
func (m *Matrix) SetScale(sx, sy float32) {
	C.MNN_Matrix_setScale(m.UnsafeC(), C.float(sx), C.float(sy))
}

// SetScaleWithPivot 设置缩放（绕指定点）
func (m *Matrix) SetScaleWithPivot(sx, sy, px, py float32) {
	C.MNN_Matrix_setScaleWithPivot(m.UnsafeC(), C.float(sx), C.float(sy), C.float(px), C.float(py))
}

// SetRotate 设置旋转（绕原点）
func (m *Matrix) SetRotate(degrees float32) {
	C.MNN_Matrix_setRotate(m.UnsafeC(), C.float(degrees))
}

// SetRotateWithPivot 设置旋转（绕指定点）
func (m *Matrix) SetRotateWithPivot(degrees, px, py float32) {
	C.MNN_Matrix_setRotateWithPivot(m.UnsafeC(), C.float(degrees), C.float(px), C.float(py))
}

// SetSinCos 设置旋转（sin/cos，绕原点）
func (m *Matrix) SetSinCos(sinValue, cosValue float32) {
	C.MNN_Matrix_setSinCos(m.UnsafeC(), C.float(sinValue), C.float(cosValue))
}

// SetSinCosWithPivot 设置旋转（sin/cos，绕指定点）
func (m *Matrix) SetSinCosWithPivot(sinValue, cosValue, px, py float32) {
	C.MNN_Matrix_setSinCosWithPivot(m.UnsafeC(), C.float(sinValue), C.float(cosValue), C.float(px), C.float(py))
}

// SetSkew 设置斜切（绕原点）
func (m *Matrix) SetSkew(kx, ky float32) {
	C.MNN_Matrix_setSkew(m.UnsafeC(), C.float(kx), C.float(ky))
}

// SetSkewWithPivot 设置斜切（绕指定点）
func (m *Matrix) SetSkewWithPivot(kx, ky, px, py float32) {
	C.MNN_Matrix_setSkewWithPivot(m.UnsafeC(), C.float(kx), C.float(ky), C.float(px), C.float(py))
}

// SetScaleTranslate 设置缩放平移
func (m *Matrix) SetScaleTranslate(sx, sy, tx, ty float32) {
	C.MNN_Matrix_setScaleTranslate(m.UnsafeC(), C.float(sx), C.float(sy), C.float(tx), C.float(ty))
}

// -------------------------- 矩阵拼接 --------------------------

// SetConcat 设置为a*b的结果
func (m *Matrix) SetConcat(a, b *Matrix) {
	C.MNN_Matrix_setConcat(m.UnsafeC(), a.UnsafeC(), b.UnsafeC())
}

// PreTranslate 前乘平移
func (m *Matrix) PreTranslate(dx, dy float32) {
	C.MNN_Matrix_preTranslate(m.UnsafeC(), C.float(dx), C.float(dy))
}

// PreScale 前乘缩放（绕原点）
func (m *Matrix) PreScale(sx, sy float32) {
	C.MNN_Matrix_preScale(m.UnsafeC(), C.float(sx), C.float(sy))
}

// PreScaleWithPivot 前乘缩放（绕指定点）
func (m *Matrix) PreScaleWithPivot(sx, sy, px, py float32) {
	C.MNN_Matrix_preScaleWithPivot(m.UnsafeC(), C.float(sx), C.float(sy), C.float(px), C.float(py))
}

// PreRotate 前乘旋转（绕原点）
func (m *Matrix) PreRotate(degrees float32) {
	C.MNN_Matrix_preRotate(m.UnsafeC(), C.float(degrees))
}

// PreRotateWithPivot 前乘旋转（绕指定点）
func (m *Matrix) PreRotateWithPivot(degrees, px, py float32) {
	C.MNN_Matrix_preRotateWithPivot(m.UnsafeC(), C.float(degrees), C.float(px), C.float(py))
}

// PreSkew 前乘斜切（绕原点）
func (m *Matrix) PreSkew(kx, ky float32) {
	C.MNN_Matrix_preSkew(m.UnsafeC(), C.float(kx), C.float(ky))
}

// PreSkewWithPivot 前乘斜切（绕指定点）
func (m *Matrix) PreSkewWithPivot(kx, ky, px, py float32) {
	C.MNN_Matrix_preSkewWithPivot(m.UnsafeC(), C.float(kx), C.float(ky), C.float(px), C.float(py))
}

// PreConcat 前乘另一个矩阵
func (m *Matrix) PreConcat(other *Matrix) {
	C.MNN_Matrix_preConcat(m.UnsafeC(), other.UnsafeC())
}

// PostTranslate 后乘平移
func (m *Matrix) PostTranslate(dx, dy float32) {
	C.MNN_Matrix_postTranslate(m.UnsafeC(), C.float(dx), C.float(dy))
}

// PostScale 后乘缩放（绕原点）
func (m *Matrix) PostScale(sx, sy float32) {
	C.MNN_Matrix_postScale(m.UnsafeC(), C.float(sx), C.float(sy))
}

// PostScaleWithPivot 后乘缩放（绕指定点）
func (m *Matrix) PostScaleWithPivot(sx, sy, px, py float32) {
	C.MNN_Matrix_postScaleWithPivot(m.UnsafeC(), C.float(sx), C.float(sy), C.float(px), C.float(py))
}

// PostIDiv 后乘逆缩放
func (m *Matrix) PostIDiv(divx, divy int) bool {
	return bool(C.MNN_Matrix_postIDiv(m.UnsafeC(), C.int(divx), C.int(divy)))
}

// PostRotate 后乘旋转（绕原点）
func (m *Matrix) PostRotate(degrees float32) {
	C.MNN_Matrix_postRotate(m.UnsafeC(), C.float(degrees))
}

// PostRotateWithPivot 后乘旋转（绕指定点）
func (m *Matrix) PostRotateWithPivot(degrees, px, py float32) {
	C.MNN_Matrix_postRotateWithPivot(m.UnsafeC(), C.float(degrees), C.float(px), C.float(py))
}

// PostSkew 后乘斜切（绕原点）
func (m *Matrix) PostSkew(kx, ky float32) {
	C.MNN_Matrix_postSkew(m.UnsafeC(), C.float(kx), C.float(ky))
}

// PostSkewWithPivot 后乘斜切（绕指定点）
func (m *Matrix) PostSkewWithPivot(kx, ky, px, py float32) {
	C.MNN_Matrix_postSkewWithPivot(m.UnsafeC(), C.float(kx), C.float(ky), C.float(px), C.float(py))
}

// PostConcat 后乘另一个矩阵
func (m *Matrix) PostConcat(other *Matrix) {
	C.MNN_Matrix_postConcat(m.UnsafeC(), other.UnsafeC())
}

// -------------------------- 变换应用 --------------------------

// SetRectToRect 设置矩形到矩形的映射
func (m *Matrix) SetRectToRect(src, dst Rect, stf MNN_Matrix_ScaleToFit) bool {
	cSrc := C.MNN_Rect{
		left:   C.float(src.Left),
		top:    C.float(src.Top),
		right:  C.float(src.Right),
		bottom: C.float(src.Bottom),
	}
	cDst := C.MNN_Rect{
		left:   C.float(dst.Left),
		top:    C.float(dst.Top),
		right:  C.float(dst.Right),
		bottom: C.float(dst.Bottom),
	}
	return bool(C.MNN_Matrix_setRectToRect(m.UnsafeC(), &cSrc, &cDst, C.MNN_Matrix_ScaleToFit(stf)))
}

// SetPolyToPoly 设置多点到多点的映射
func (m *Matrix) SetPolyToPoly(src, dst []Point) bool {
	if len(src) != len(dst) || len(src) > 4 {
		return false
	}
	count := len(src)
	cSrc := make([]C.MNN_Point, count)
	cDst := make([]C.MNN_Point, count)
	for i := 0; i < count; i++ {
		cSrc[i] = C.MNN_Point{x: C.float(src[i].X), y: C.float(src[i].Y)}
		cDst[i] = C.MNN_Point{x: C.float(dst[i].X), y: C.float(dst[i].Y)}
	}
	return bool(C.MNN_Matrix_setPolyToPoly(m.UnsafeC(), &cSrc[0], &cDst[0], C.int(count)))
}

// Invert 矩阵求逆
func (m *Matrix) Invert() (*Matrix, error) {
	inverse := NewMatrix()
	ok := C.MNN_Matrix_invert(m.UnsafeC(), inverse.UnsafeC())
	if !bool(ok) {
		inverse.Close()
		return nil, errors.New("matrix is not invertible")
	}
	return inverse, nil
}

// SetAffineIdentity 设置仿射矩阵为单位矩阵
func SetAffineIdentity(affine [6]float32) {
	var cAffine [6]C.float
	for i := 0; i < 6; i++ {
		cAffine[i] = C.float(affine[i])
	}
	C.MNN_Matrix_SetAffineIdentity(&cAffine[0])
	for i := 0; i < 6; i++ {
		affine[i] = float32(cAffine[i])
	}
}

// AsAffine 转换为仿射矩阵
func (m *Matrix) AsAffine() ([6]float32, bool) {
	var cAffine [6]C.float
	ok := C.MNN_Matrix_asAffine(m.UnsafeC(), &cAffine[0])
	var affine [6]float32
	for i := 0; i < 6; i++ {
		affine[i] = float32(cAffine[i])
	}
	return affine, bool(ok)
}

// SetAffine 从仿射矩阵设置
func (m *Matrix) SetAffine(affine [6]float32) {
	var cAffine [6]C.float
	for i := 0; i < 6; i++ {
		cAffine[i] = C.float(affine[i])
	}
	C.MNN_Matrix_setAffine(m.UnsafeC(), &cAffine[0])
}

// MapPoints 映射点数组
func (m *Matrix) MapPoints(src []Point) (dst []Point) {
	/*if len(src) == 0 {
		return nil
	}
	count := len(src)
	cSrc := make([]C.MNN_Point, count)
	for i := 0; i < count; i++ {
		cSrc[i] = C.MNN_Point{x: C.float(src[i].X), y: C.float(src[i].Y)}
	}
	cDst := make([]C.MNN_Point, count)
	C.MNN_Matrix_mapPoints(m.UnsafeC(), &cDst[0], &cSrc[0], C.int(count))
	dst := make([]Point, count)
	for i := 0; i < count; i++ {
		dst[i] = Point{X: float32(cDst[i].x), Y: float32(cDst[i].y)}
	}
	return dst*/

	for i, p := range src {
		dst[i] = m.MapXY(p.X, p.Y)
	}
	return dst
}

// MapPointsInPlace 原地映射点数组
func (m *Matrix) MapPointsInPlace(pts []Point) {
	/*if len(pts) == 0 {
		return
	}
	count := len(pts)
	cPts := make([]C.MNN_Point, count)
	for i := 0; i < count; i++ {
		cPts[i] = C.MNN_Point{x: C.float(pts[i].X), y: C.float(pts[i].Y)}
	}
	C.MNN_Matrix_mapPointsInPlace(m.UnsafeC(), &cPts[0], C.int(count))
	for i := 0; i < count; i++ {
		pts[i] = Point{X: float32(cPts[i].x), Y: float32(cPts[i].y)}
	}*/

	// 直接遍历切片下标，原地修改原切片的每个元素
	for i := range pts {
		// 对 pts[i] 执行矩阵坐标变换，结果直接写回原位置
		pts[i] = m.MapXY(pts[i].X, pts[i].Y) // 替换成你实际的单点映射方法
	}
}

// MapXY 映射单个点
func (m *Matrix) MapXY(x, y float32) Point {
	/*cP := C.MNN_Matrix_mapXY(m.UnsafeC(), C.float(x), C.float(y))
	return Point{X: float32(cP.x), Y: float32(cP.y)}*/

	mat := m.fMat
	// 齐次坐标计算
	w := mat[6]*x + mat[7]*y + mat[8]
	if w == 0 {
		return Point{0, 0}
	}

	nx := (mat[0]*x + mat[1]*y + mat[2]) / w
	ny := (mat[3]*x + mat[4]*y + mat[5]) / w
	return Point{nx, ny}
}

// MapRect 映射矩形
func (m *Matrix) MapRect(src Rect) (Rect, bool) {
	cSrc := C.MNN_Rect{
		left:   C.float(src.Left),
		top:    C.float(src.Top),
		right:  C.float(src.Right),
		bottom: C.float(src.Bottom),
	}
	var cDst C.MNN_Rect
	ok := C.MNN_Matrix_mapRect(m.UnsafeC(), &cDst, &cSrc)
	dst := Rect{
		Left:   float32(cDst.left),
		Top:    float32(cDst.top),
		Right:  float32(cDst.right),
		Bottom: float32(cDst.bottom),
	}
	return dst, bool(ok)
}

// MapRectInPlace 原地映射矩形
func (m *Matrix) MapRectInPlace(rect *Rect) bool {
	cRect := C.MNN_Rect{
		left:   C.float(rect.Left),
		top:    C.float(rect.Top),
		right:  C.float(rect.Right),
		bottom: C.float(rect.Bottom),
	}
	ok := C.MNN_Matrix_mapRectInPlace(m.UnsafeC(), &cRect)
	rect.Left = float32(cRect.left)
	rect.Top = float32(cRect.top)
	rect.Right = float32(cRect.right)
	rect.Bottom = float32(cRect.bottom)
	return bool(ok)
}

// MapRectScaleTranslate 仅缩放平移模式下映射矩形
func (m *Matrix) MapRectScaleTranslate(src Rect) Rect {
	cSrc := C.MNN_Rect{
		left:   C.float(src.Left),
		top:    C.float(src.Top),
		right:  C.float(src.Right),
		bottom: C.float(src.Bottom),
	}
	var cDst C.MNN_Rect
	C.MNN_Matrix_mapRectScaleTranslate(m.UnsafeC(), &cDst, &cSrc)
	return Rect{
		Left:   float32(cDst.left),
		Top:    float32(cDst.top),
		Right:  float32(cDst.right),
		Bottom: float32(cDst.bottom),
	}
}

// -------------------------- 比较/调试 --------------------------

// CheapEqualTo 快速比较（内存逐字节）
func (m *Matrix) CheapEqualTo(other *Matrix) bool {
	return bool(C.MNN_Matrix_cheapEqualTo(m.UnsafeC(), other.UnsafeC()))
}

// Equals 数值比较（==）
func (m *Matrix) Equals(other *Matrix) bool {
	return bool(C.MNN_Matrix_equals(m.UnsafeC(), other.UnsafeC()))
}

// NotEquals 数值比较（!=）
func (m *Matrix) NotEquals(other *Matrix) bool {
	return bool(C.MNN_Matrix_notEquals(m.UnsafeC(), other.UnsafeC()))
}

// Dump 打印矩阵信息
func (m *Matrix) Dump() {
	//C.MNN_Matrix_dump(m.UnsafeC())
	fmt.Printf(m.String())
}

// -------------------------- 缩放因子 --------------------------

// GetMinScale 获取最小缩放因子
func (m *Matrix) GetMinScale() float32 {
	//return float32(C.MNN_Matrix_getMinScale(m.UnsafeC()))
	var scales [2]float32
	var ok bool
	if scales, ok = m.GetMinMaxScales(); !ok {
		return -1
	}
	return scales[0]
}

// GetMaxScale 获取最大缩放因子
func (m *Matrix) GetMaxScale() float32 {
	//return float32(C.MNN_Matrix_getMaxScale(m.UnsafeC()))
	var scales [2]float32
	var ok bool
	if scales, ok = m.GetMinMaxScales(); !ok {
		return -1
	}
	return scales[1]
}

// GetMinMaxScales 获取最小/最大缩放因子
func (m *Matrix) GetMinMaxScales() (scales [2]float32, ok bool) {
	/*var cScales [2]C.float
	ok := C.MNN_Matrix_getMinMaxScales(m.UnsafeC(), &cScales[0])
	var scales [2]float32
	scales[0] = float32(cScales[0])
	scales[1] = float32(cScales[1])
	return scales, bool(ok)*/

	ok = false
	// 分解仿射矩阵为缩放+旋转
	a := m.GetScaleX()
	b := m.GetSkewX()
	c := m.GetSkewY()
	d := m.GetScaleY()

	// 计算奇异值
	// 参考：https://en.wikipedia.org/wiki/Singular_value_decomposition
	// 2x2矩阵的奇异值 = sqrt( (tr(A^T A) ± sqrt(tr(A^T A)^2 - 4 det(A^T A)) ) / 2 )
	aa := a*a + c*c
	bb := a*b + c*d
	dd := b*b + d*d
	tr := aa + dd
	det := aa*dd - bb*bb

	if det < 0 {
		return
	}

	sqrtDet := math.Sqrt(float64(det))
	s1 := math.Sqrt((float64(tr) + 2*sqrtDet) / 2)
	s2 := math.Sqrt((float64(tr) - 2*sqrtDet) / 2)

	scales[0] = float32(math.Min(s1, s2))
	scales[1] = float32(math.Max(s1, s2))

	ok = true
	return
}

// -------------------------- 工具函数 --------------------------

// GetIdentity 获取单位矩阵常量
func GetIdentity() *Matrix {
	/*cMat := C.MNN_Matrix_GetIdentity()
	return &Matrix{cMatrix: cMat, owner: false} // 不拥有内存*/
	m := NewMatrix()
	m.Reset()
	return m
}

// GetInvalidMatrix 获取无效矩阵常量
func GetInvalidMatrix() *Matrix {
	/*cMat := C.MNN_Matrix_GetInvalidMatrix()
	return &Matrix{cMatrix: cMat, owner: false} // 不拥有内存*/

	m := NewMatrix()
	for i := range m.fMat {
		m.fMat[i] = math.MaxFloat32
	}
	return m
}

// Concat 创建矩阵拼接结果（a*b）
func Concat(a, b *Matrix) *Matrix {
	cMat := C.MNN_Matrix_Concat(a.UnsafeC(), b.UnsafeC())
	if cMat == nil {
		panic("failed to concat matrices")
	}
	return FromCMatrix(cMat)
}

// DirtyMatrixTypeCache 标记矩阵类型缓存为脏
func (m *Matrix) DirtyMatrixTypeCache() {
	C.MNN_Matrix_dirtyMatrixTypeCache(m.UnsafeC())
}

// String 矩阵的字符串表示
/*func (m *Matrix) String() string {
	elems := m.Get9()
	return fmt.Sprintf(`MNN_Matrix[
	%6.2f, %6.2f, %6.2f
	%6.2f, %6.2f, %6.2f
	%6.2f, %6.2f, %6.2f
]`,
		elems[0], elems[1], elems[2],
		elems[3], elems[4], elems[5],
		elems[6], elems[7], elems[8],
	)
}*/
func (m *Matrix) String() string {
	return fmt.Sprintf(`MNN_Matrix[
	%6.2f, %6.2f, %6.2f
	%6.2f, %6.2f, %6.2f
	%6.2f, %6.2f, %6.2f
]`,
		m.fMat[0], m.fMat[1], m.fMat[2],
		m.fMat[3], m.fMat[4], m.fMat[5],
		m.fMat[6], m.fMat[7], m.fMat[8],
	)
}
