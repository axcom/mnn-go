package mnn

/*
#include "Matrix_c.h"
#include <stdlib.h>
*/
import "C"
import (
	"math"
	"unsafe"
)

// 对应C中的SK_ScalarMin/SK_ScalarMax（模拟Skia的标量极值）
const (
	SKScalarMin = -math.MaxFloat32
	SKScalarMax = math.MaxFloat32
)

// Point 二维点结构体，对应C的Point
type Point struct {
	X float32
	Y float32
}

// Rect 矩形结构体，与C的Rect内存布局对齐
type Rect struct {
	Left   float32 // 左边界
	Top    float32 // 上边界
	Right  float32 // 右边界
	Bottom float32 // 下边界
}

// ===================== Point 方法 =====================

// C++ -> Go 转换
func FromCPoint(cPoint *C.MNN_Point) *Point {
	return (*Point)(unsafe.Pointer(cPoint))
}

// Go -> C++ 转换: 在确保内存布局一致，通过unsafe转换
func (r *Point) UnsafeC() *C.MNN_Point {
	return (*C.MNN_Point)(unsafe.Pointer(r))
}

// PointSet 设置点的坐标
func PointSet(point *Point, x, y float32) {
	if point == nil {
		return
	}
	point.X = x
	point.Y = y
}

// ===================== Rect 构造函数 =====================

// RectMakeEmpty 创建空矩形
func RectMakeEmpty() Rect {
	return Rect{0, 0, 0, 0}
}

// RectMakeLargest 创建最大范围的矩形
func RectMakeLargest() Rect {
	return Rect{SKScalarMin, SKScalarMin, SKScalarMax, SKScalarMax}
}

// RectMakeWH 从宽高创建矩形（左上角为原点）
func RectMakeWH(w, h float32) Rect {
	return Rect{0, 0, w, h}
}

// RectMakeIWH 从整型宽高创建矩形
func RectMakeIWH(w, h int) Rect {
	var rect Rect
	RectSet(&rect, 0, 0, float32(w), float32(h))
	return rect
}

// RectMakeLTRB 从左、上、右、下创建矩形
func RectMakeLTRB(l, t, r, b float32) Rect {
	return Rect{l, t, r, b}
}

// RectMakeXYWH 从x、y、宽、高创建矩形
func RectMakeXYWH(x, y, w, h float32) Rect {
	return Rect{x, y, x + w, y + h}
}

// C++ -> Go 转换
func FromCRect(cRect *C.MNN_Rect) *Rect {
	return (*Rect)(unsafe.Pointer(cRect))
}

// Go -> C++ 转换: 在确保内存布局一致，通过unsafe转换
func (r *Rect) UnsafeC() *C.MNN_Rect {
	return (*C.MNN_Rect)(unsafe.Pointer(r))
}

// ===================== Rect 属性查询方法 =====================

// IsEmpty 判断矩形是否为空（左>=右 或 上>=下）
func (r *Rect) IsEmpty() bool {
	return !(r.Left < r.Right && r.Top < r.Bottom)
}

// IsSorted 判断矩形的边界是否有序（左<=右 且 上<=下）
func (r *Rect) IsSorted() bool {
	return r.Left <= r.Right && r.Top <= r.Bottom
}

// X 获取矩形的X坐标（左边界）
func (r *Rect) X() float32 {
	return r.Left
}

// Y 获取矩形的Y坐标（上边界）
func (r *Rect) Y() float32 {
	return r.Top
}

// Width 获取矩形宽度
func (r *Rect) Width() float32 {
	return r.Right - r.Left
}

// Height 获取矩形高度
func (r *Rect) Height() float32 {
	return r.Bottom - r.Top
}

// CenterX 获取矩形中心X坐标
func (r *Rect) CenterX() float32 {
	return 0.5*r.Left + 0.5*r.Right // 避免溢出
}

// CenterY 获取矩形中心Y坐标
func (r *Rect) CenterY() float32 {
	return 0.5*r.Top + 0.5*r.Bottom
}

// ===================== Rect 设置操作方法 =====================

// SetEmpty 将矩形设为空
func (r *Rect) SetEmpty() {
	*r = RectMakeEmpty()
}

// RectSet 设置矩形的四个边界
func RectSet(r *Rect, left, top, right, bottom float32) {
	r.Left = left
	r.Top = top
	r.Right = right
	r.Bottom = bottom
}

// SetLTRB 等价于RectSet
func (r *Rect) SetLTRB(left, top, right, bottom float32) {
	RectSet(r, left, top, right, bottom)
}

// ISet 用整型值设置矩形边界
func (r *Rect) ISet(left, top, right, bottom int) {
	r.Left = float32(left)
	r.Top = float32(top)
	r.Right = float32(right)
	r.Bottom = float32(bottom)
}

// ISetWH 用整型宽高设置矩形（左上角为原点）
func (r *Rect) ISetWH(width, height int) {
	r.Left = 0
	r.Top = 0
	r.Right = float32(width)
	r.Bottom = float32(height)
}

// SetXYWH 从x、y、宽、高设置矩形
func (r *Rect) SetXYWH(x, y, width, height float32) {
	r.Left = x
	r.Top = y
	r.Right = x + width
	r.Bottom = y + height
}

// SetWH 从宽高设置矩形（左上角为原点）
func (r *Rect) SetWH(width, height float32) {
	r.Left = 0
	r.Top = 0
	r.Right = width
	r.Bottom = height
}

// ===================== Rect 几何变换方法 =====================

// MakeOffset 创建偏移后的新矩形
func (r *Rect) MakeOffset(dx, dy float32) Rect {
	return RectMakeLTRB(
		r.Left+dx,
		r.Top+dy,
		r.Right+dx,
		r.Bottom+dy,
	)
}

// MakeInset 创建内缩后的新矩形
func (r *Rect) MakeInset(dx, dy float32) Rect {
	return RectMakeLTRB(
		r.Left+dx,
		r.Top+dy,
		r.Right-dx,
		r.Bottom-dy,
	)
}

// MakeOutset 创建外扩后的新矩形
func (r *Rect) MakeOutset(dx, dy float32) Rect {
	return RectMakeLTRB(
		r.Left-dx,
		r.Top-dy,
		r.Right+dx,
		r.Bottom+dy,
	)
}

// Offset 偏移矩形（原地修改）
func (r *Rect) Offset(dx, dy float32) {
	r.Left += dx
	r.Top += dy
	r.Right += dx
	r.Bottom += dy
}

// OffsetTo 偏移矩形到指定坐标（原地修改）
func (r *Rect) OffsetTo(newX, newY float32) {
	r.Right += newX - r.Left
	r.Bottom += newY - r.Top
	r.Left = newX
	r.Top = newY
}

// Inset 内缩矩形（原地修改）
func (r *Rect) Inset(dx, dy float32) {
	r.Left += dx
	r.Top += dy
	r.Right -= dx
	r.Bottom -= dy
}

// Outset 外扩矩形（原地修改）
func (r *Rect) Outset(dx, dy float32) {
	r.Inset(-dx, -dy)
}

// ===================== Rect 相交/合并方法 =====================

// 内部方法：判断两个LTRB区域是否相交
func intersectsImpl(al, at, ar, ab, bl, bt, br, bb float32) bool {
	L := maxFloat32(al, bl)
	R := minFloat32(ar, br)
	T := maxFloat32(at, bt)
	B := minFloat32(ab, bb)
	return L < R && T < B
}

// IntersectsLTRB 判断当前矩形与指定LTRB区域是否相交
func (r *Rect) IntersectsLTRB(left, top, right, bottom float32) bool {
	return intersectsImpl(r.Left, r.Top, r.Right, r.Bottom, left, top, right, bottom)
}

// Intersects 判断当前矩形与另一个矩形是否相交
func (r *Rect) Intersects(other *Rect) bool {
	if other == nil {
		return false
	}
	return intersectsImpl(r.Left, r.Top, r.Right, r.Bottom, other.Left, other.Top, other.Right, other.Bottom)
}

// JoinNonEmptyArg 合并非空矩形（若当前矩形为空，则直接赋值）
func (r *Rect) JoinNonEmptyArg(other *Rect) {
	if other == nil {
		return
	}
	if other.IsEmpty() {
		return // 模拟C的MNN_ASSERT，这里直接返回
	}
	if r.IsEmpty() {
		*r = *other
	} else {
		r.JoinPossiblyEmptyRect(other)
	}
}

// JoinPossiblyEmptyRect 合并两个矩形（允许空矩形）
func (r *Rect) JoinPossiblyEmptyRect(other *Rect) {
	if other == nil {
		return
	}
	r.Left = minFloat32(r.Left, other.Left)
	r.Top = minFloat32(r.Top, other.Top)
	r.Right = maxFloat32(r.Right, other.Right)
	r.Bottom = maxFloat32(r.Bottom, other.Bottom)
}

// Contains 判断点(x,y)是否在矩形内
func (r *Rect) Contains(x, y float32) bool {
	if r.IsEmpty() {
		return false
	}
	return x >= r.Left && x < r.Right && y >= r.Top && y < r.Bottom
}

// ===================== Rect 排序方法 =====================

// Sort 对矩形边界进行排序（左<右，上<下，原地修改）
func (r *Rect) Sort() {
	// 交换左右
	if r.Left > r.Right {
		r.Left, r.Right = r.Right, r.Left
	}
	// 交换上下
	if r.Top > r.Bottom {
		r.Top, r.Bottom = r.Bottom, r.Top
	}
}

// MakeSorted 创建排序后的新矩形
func (r *Rect) MakeSorted() Rect {
	left := minFloat32(r.Left, r.Right)
	top := minFloat32(r.Top, r.Bottom)
	right := maxFloat32(r.Left, r.Right)
	bottom := maxFloat32(r.Top, r.Bottom)
	return RectMakeLTRB(left, top, right, bottom)
}

// AsScalars 将矩形转换为float32切片（对应C的指针数组）
func (r *Rect) AsScalars() []float32 {
	return []float32{r.Left, r.Top, r.Right, r.Bottom}
}

// ===================== 工具函数 =====================

// minFloat32 获取两个float32的最小值
func minFloat32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

// maxFloat32 获取两个float32的最大值
func maxFloat32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
