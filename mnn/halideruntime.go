package mnn

/*
#include <stdlib.h>
#include "MNN/HalideRuntime.h"
typedef struct halide_type_t halide_type_t;
void setBufferType(halide_buffer_t* buffer, halide_type_t* inType){
	buffer->type = *inType;
};
halide_type_t* getBufferType(halide_buffer_t* buffer){
	return &(buffer->type);
};
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// -------------------------- 枚举类型映射 --------------------------

// HalideTypeCode Halide类型系统的基础类型编码
type HalideTypeCode int

const (
	HalideType_Int    HalideTypeCode = C.halide_type_int    // 有符号整数
	HalideType_UInt   HalideTypeCode = C.halide_type_uint   // 无符号整数
	HalideType_Float  HalideTypeCode = C.halide_type_float  // IEEE浮点数
	HalideType_Handle HalideTypeCode = C.halide_type_handle // 不透明指针类型（void*）
	HalideType_BFloat HalideTypeCode = C.halide_type_bfloat // bfloat格式浮点数
)

// HalideBufferFlag Buffer标志位
type HalideBufferFlag uint64

const (
	HalideBufferFlagHostDirty   HalideBufferFlag = C.halide_buffer_flag_host_dirty   // 主机内存数据脏
	HalideBufferFlagDeviceDirty HalideBufferFlag = C.halide_buffer_flag_device_dirty // 设备内存数据脏
)

// -------------------------- 核心结构体封装 --------------------------

// HalideType Halide类型系统的运行时标签（对应C的halide_type_t）
type HalideType struct {
	Code  HalideTypeCode // 基础类型编码
	Bits  uint8          // 单个标量值的精度位数
	Lanes uint16         // 向量元素个数（标量为1）
}

// HalideDimension 维度信息（对应C的halide_dimension_t）
type HalideDimension struct {
	Min    int32  // 维度最小值
	Extent int32  // 维度长度
	Stride int32  // 维度步长
	Flags  uint32 // 维度标志（预留）
}

// HalideDeviceInterface GPU设备接口（对应C的halide_device_interface_t）
// 注：函数指针字段在Go侧仅用于传递，不建议直接调用（需绑定具体GPU实现）
type HalideDeviceInterface struct {
	impl *C.struct_halide_device_interface_t // C侧接口句柄
}

// HalideBuffer Halide Buffer的Go封装（对应C的halide_buffer_t）
type HalideBuffer struct {
	cBuf    *C.struct_halide_buffer_t // C侧Buffer句柄
	ownHost bool                      // 是否拥有host内存（自动释放）
	ownDim  bool                      // 是否拥有dim数组内存（自动释放）
}

// -------------------------- HalideType 方法 --------------------------

// NewHalideType 创建HalideType实例
func NewHalideType(code HalideTypeCode, bits uint8, lanes uint16) HalideType {
	return HalideType{
		Code:  code,
		Bits:  bits,
		Lanes: lanes,
	}
}

// NewScalarHalideType 创建标量类型的HalideType（lanes=1）
func NewScalarHalideType(code HalideTypeCode, bits uint8) HalideType {
	return NewHalideType(code, bits, 1)
}

// ToCType 转换为C侧halide_type_t
func (t HalideType) ToCType() C.struct_halide_type_t {
	return C.struct_halide_type_t{
		code:  C.halide_type_code_t(t.Code),
		bits:  C.uint8_t(t.Bits),
		lanes: C.uint16_t(t.Lanes),
	}
}

// FromCType 从C侧halide_type_t转换为Go类型
func FromCType(cType *C.struct_halide_type_t) HalideType {
	if cType == nil {
		return HalideType{}
	}
	return HalideType{
		Code:  HalideTypeCode(cType.code),
		Bits:  uint8(cType.bits),
		Lanes: uint16(cType.lanes),
	}
}

// Bytes 计算单个元素的字节大小（替代C的halide_type_bytes）
func (t HalideType) Bytes() int {
	return (int(t.Bits) + 7) / 8
}

// Equal 比较两个HalideType是否相等（替代C的halide_type_equal）
func (t HalideType) Equal(other HalideType) bool {
	return t.Code == other.Code && t.Bits == other.Bits && t.Lanes == other.Lanes
}

// -------------------------- 便捷类型创建函数 --------------------------

// HalideTypeFloat32 创建float32标量类型
func HalideTypeFloat32() HalideType {
	return NewScalarHalideType(HalideType_Float, 32)
}

// HalideTypeFloat64 创建float64标量类型
func HalideTypeFloat64() HalideType {
	return NewScalarHalideType(HalideType_Float, 64)
}

// HalideTypeBool 创建bool标量类型（uint1）
func HalideTypeBool() HalideType {
	return NewScalarHalideType(HalideType_UInt, 1)
}

// HalideTypeUInt8 创建uint8标量类型
func HalideTypeUInt8() HalideType {
	return NewScalarHalideType(HalideType_UInt, 8)
}

// HalideTypeUInt16 创建uint16标量类型
func HalideTypeUInt16() HalideType {
	return NewScalarHalideType(HalideType_UInt, 16)
}

// HalideTypeUInt32 创建uint32标量类型
func HalideTypeUInt32() HalideType {
	return NewScalarHalideType(HalideType_UInt, 32)
}

// HalideTypeUInt64 创建uint64标量类型
func HalideTypeUInt64() HalideType {
	return NewScalarHalideType(HalideType_UInt, 64)
}

// HalideTypeInt8 创建int8标量类型
func HalideTypeInt8() HalideType {
	return NewScalarHalideType(HalideType_Int, 8)
}

// HalideTypeInt16 创建int16标量类型
func HalideTypeInt16() HalideType {
	return NewScalarHalideType(HalideType_Int, 16)
}

// HalideTypeInt32 创建int32标量类型
func HalideTypeInt32() HalideType {
	return NewScalarHalideType(HalideType_Int, 32)
}

// HalideTypeInt64 创建int64标量类型
func HalideTypeInt64() HalideType {
	return NewScalarHalideType(HalideType_Int, 64)
}

// HalideTypeHandle 创建handle类型（void*，64位）
func HalideTypeHandle() HalideType {
	return NewScalarHalideType(HalideType_Handle, 64)
}

// HalideTypeBFloat 创建bfloat16标量类型
func HalideTypeBFloat() HalideType {
	return NewScalarHalideType(HalideType_BFloat, 16)
}

// -------------------------- HalideDimension 方法 --------------------------

// ToCType 转换为C侧halide_dimension_t
func (d HalideDimension) ToCType() C.struct_halide_dimension_t {
	return C.struct_halide_dimension_t{
		min:    C.int32_t(d.Min),
		extent: C.int32_t(d.Extent),
		stride: C.int32_t(d.Stride),
		flags:  C.uint32_t(d.Flags),
	}
}

// FromCType 从C侧halide_dimension_t转换为Go类型
func FromCTypeDimension(cDim C.struct_halide_dimension_t) HalideDimension {
	return HalideDimension{
		Min:    int32(cDim.min),
		Extent: int32(cDim.extent),
		Stride: int32(cDim.stride),
		Flags:  uint32(cDim.flags),
	}
}

// Equal 比较两个HalideDimension是否相等
func (d HalideDimension) Equal(other HalideDimension) bool {
	return d.Min == other.Min && d.Extent == other.Extent && d.Stride == other.Stride && d.Flags == other.Flags
}

// -------------------------- HalideBuffer 核心方法 --------------------------

// NewHalideBuffer 创建HalideBuffer实例
// dimensions: 维度数
// dims: 维度信息数组（长度需等于dimensions）
// typ: 元素类型
// host: 主机内存指针（nil则自动分配内存，非nil则由用户管理）
// autoAllocHost: 是否自动分配主机内存（仅当host为nil时生效）
func NewHalideBuffer(dimensions int, dims []HalideDimension, typ HalideType, host unsafe.Pointer, autoAllocHost bool) (*HalideBuffer, error) {
	if dimensions <= 0 || len(dims) != dimensions {
		return nil, fmt.Errorf("维度数与维度数组长度不匹配")
	}

	// 分配C侧Buffer结构体
	cBuf := (*C.struct_halide_buffer_t)(C.malloc(C.sizeof_struct_halide_buffer_t))
	if cBuf == nil {
		return nil, fmt.Errorf("分配C侧Buffer失败")
	}

	// 初始化C侧Buffer默认值
	*cBuf = C.struct_halide_buffer_t{
		device:           0,
		device_interface: nil,
		host:             (*C.uint8_t)(host),
		flags:            0,
		//type_:            typ.ToCType(),
		dimensions: C.int32_t(dimensions),
		dim:        nil,
		padding:    nil,
	}
	cType := typ.ToCType()
	C.setBufferType(cBuf, &cType)

	// 分配维度数组内存
	cDims := (*C.struct_halide_dimension_t)(C.malloc(C.size_t(dimensions) * C.sizeof_struct_halide_dimension_t))
	if cDims == nil {
		C.free(unsafe.Pointer(cBuf))
		return nil, fmt.Errorf("分配维度数组失败")
	}
	// 拷贝维度信息到C侧数组
	cDimSlice := unsafe.Slice(cDims, dimensions)
	for i, dim := range dims {
		cDimSlice[i] = dim.ToCType()
	}
	cBuf.dim = cDims

	// 自动分配主机内存
	ownHost := false
	if host == nil && autoAllocHost {
		// 计算总字节数：各维度extent乘积 * 单个元素字节数
		totalElements := 1
		for _, dim := range dims {
			totalElements *= int(dim.Extent)
		}
		totalBytes := totalElements * typ.Bytes()
		if totalBytes > 0 {
			cBuf.host = (*C.uint8_t)(C.malloc(C.size_t(totalBytes)))
			if cBuf.host == nil {
				C.free(unsafe.Pointer(cDims))
				C.free(unsafe.Pointer(cBuf))
				return nil, fmt.Errorf("分配主机内存失败")
			}
			ownHost = true
		}
	}

	buf := &HalideBuffer{
		cBuf:    cBuf,
		ownHost: ownHost,
		ownDim:  true,
	}

	// 注册Finalizer自动释放内存
	runtime.SetFinalizer(buf, func(b *HalideBuffer) {
		b.Destroy()
	})

	return buf, nil
}

// Destroy 手动销毁HalideBuffer（释放C侧内存）
func (b *HalideBuffer) Destroy() {
	if b == nil || b.cBuf == nil {
		return
	}

	// 释放主机内存（仅当自动分配时）
	if b.ownHost && b.cBuf.host != nil {
		C.free(unsafe.Pointer(b.cBuf.host))
		b.cBuf.host = nil
	}

	// 释放维度数组内存（仅当自动分配时）
	if b.ownDim && b.cBuf.dim != nil {
		C.free(unsafe.Pointer(b.cBuf.dim))
		b.cBuf.dim = nil
	}

	// 释放Buffer结构体
	C.free(unsafe.Pointer(b.cBuf))
	b.cBuf = nil
}

// SetFlags 设置Buffer标志位
func (b *HalideBuffer) SetFlags(flags ...HalideBufferFlag) {
	if b == nil || b.cBuf == nil {
		panic("HalideBuffer未初始化或已销毁")
	}
	var cFlags C.uint64_t = 0
	for _, flag := range flags {
		cFlags |= C.uint64_t(flag)
	}
	b.cBuf.flags = cFlags
}

// GetFlags 获取Buffer标志位
func (b *HalideBuffer) GetFlags() []HalideBufferFlag {
	if b == nil || b.cBuf == nil {
		panic("HalideBuffer未初始化或已销毁")
	}
	var flags []HalideBufferFlag
	if b.cBuf.flags&C.uint64_t(HalideBufferFlagHostDirty) != 0 {
		flags = append(flags, HalideBufferFlagHostDirty)
	}
	if b.cBuf.flags&C.uint64_t(HalideBufferFlagDeviceDirty) != 0 {
		flags = append(flags, HalideBufferFlagDeviceDirty)
	}
	return flags
}

// HostPointer 获取主机内存指针
func (b *HalideBuffer) HostPointer() unsafe.Pointer {
	if b == nil || b.cBuf == nil {
		panic("HalideBuffer未初始化或已销毁")
	}
	return unsafe.Pointer(b.cBuf.host)
}

// Dimensions 获取维度数
func (b *HalideBuffer) Dimensions() int {
	if b == nil || b.cBuf == nil {
		panic("HalideBuffer未初始化或已销毁")
	}
	return int(b.cBuf.dimensions)
}

// Dims 获取维度信息数组
func (b *HalideBuffer) Dims() []HalideDimension {
	if b == nil || b.cBuf == nil {
		panic("HalideBuffer未初始化或已销毁")
	}
	dimensions := int(b.cBuf.dimensions)
	if dimensions == 0 || b.cBuf.dim == nil {
		return nil
	}
	cDimSlice := unsafe.Slice(b.cBuf.dim, dimensions)
	dims := make([]HalideDimension, dimensions)
	for i, cDim := range cDimSlice {
		dims[i] = FromCTypeDimension(cDim)
	}
	return dims
}

// Type 获取元素类型
func (b *HalideBuffer) Type() HalideType {
	if b == nil || b.cBuf == nil {
		panic("HalideBuffer未初始化或已销毁")
	}
	//return FromCType(b.cBuf.`type`)
	cType := C.getBufferType(b.cBuf)
	return FromCType(cType)
}

// ToCBuffer 获取C侧halide_buffer_t指针（供底层cgo调用）
func (b *HalideBuffer) ToCBuffer() *C.struct_halide_buffer_t {
	if b == nil || b.cBuf == nil {
		panic("HalideBuffer未初始化或已销毁")
	}
	return b.cBuf
}
