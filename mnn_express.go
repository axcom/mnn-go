package mnn

/*
#cgo CFLAGS: -I.
#cgo CXXFLAGS: -I.
#cgo linux LDFLAGS: -lMNN -lMNN_Express
#cgo windows LDFLAGS: libmnn.a -lstdc++
#include <MNN_C.h>
#include <MNN_Express.h>
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// ExpressConfig 定义MNN Express的配置
type ExpressConfig struct {
	Dynamic      bool
	ShapeMutable bool
	Rearrange    bool
	Base         *ExpressModule
}

// ToCExpressConfig 将Go的ExpressConfig转换为C的MNN_Express_Config
func (ec *ExpressConfig) ToCExpressConfig() C.MNN_Express_Config {
	var base *C.MNN_Express_Module
	if ec.Base != nil && ec.Base.IsValid {
		base = ec.Base.Handle
	}

	return C.MNN_Express_Config{
		dynamic:      toCBool(ec.Dynamic),
		shapeMutable: toCBool(ec.ShapeMutable),
		rearrange:    toCBool(ec.Rearrange),
		base:         base,
	}
}

// ExpressVARP 是MNN Express VARP的Go封装
type ExpressVARP struct {
	Handle  *C.MNN_Express_VARP
	IsValid bool
}

// NewExpressVARPFromConstFloat 创建浮点常量VARP
func NewExpressVARPFromConstFloat(data []float32, dims []int) (*ExpressVARP, error) {
	if len(data) == 0 || len(dims) == 0 {
		return nil, fmt.Errorf("data or dims is empty")
	}

	cDims := make([]C.int, len(dims))
	for i, d := range dims {
		cDims[i] = C.int(d)
	}

	varp := C.MNN_Express_VARP_createConstFloat((*C.float)(&data[0]), &cDims[0], C.int(len(dims)))
	if varp == nil {
		return nil, fmt.Errorf("failed to create const float VARP")
	}

	return &ExpressVARP{Handle: varp, IsValid: true}, nil
}

// NewExpressVARPFromConstInt 创建整数常量VARP
func NewExpressVARPFromConstInt(data []int32, dims []int) (*ExpressVARP, error) {
	if len(data) == 0 || len(dims) == 0 {
		return nil, fmt.Errorf("data or dims is empty")
	}

	cDims := make([]C.int, len(dims))
	for i, d := range dims {
		cDims[i] = C.int(d)
	}

	varp := C.MNN_Express_VARP_createConstInt((*C.int)(&data[0]), &cDims[0], C.int(len(dims)))
	if varp == nil {
		return nil, fmt.Errorf("failed to create const int VARP")
	}

	return &ExpressVARP{Handle: varp, IsValid: true}, nil
}

// Close 关闭VARP
func (v *ExpressVARP) Close() {
	if v.IsValid {
		C.MNN_Express_VARP_destroy(v.Handle)
		v.IsValid = false
	}
}

// GetFloatData 获取浮点数据
func (v *ExpressVARP) GetFloatData() []float32 {
	if !v.IsValid {
		return nil
	}

	data := C.MNN_Express_VARP_getFloatData(v.Handle)
	if data == nil {
		return nil
	}

	elementSize := int(C.MNN_Express_VARP_elementSize(v.Handle))
	return (*[1 << 30]float32)(unsafe.Pointer(data))[:elementSize]
}

// ElementSize 获取元素数量
func (v *ExpressVARP) ElementSize() int {
	if !v.IsValid {
		return 0
	}

	return int(C.MNN_Express_VARP_elementSize(v.Handle))
}

// GetShape 获取形状
func (v *ExpressVARP) GetShape() []int {
	if !v.IsValid {
		return nil
	}

	// 先获取元素数量来估算最大维度数
	elementSize := v.ElementSize()
	if elementSize == 0 {
		return nil
	}

	// 尝试获取形状
	maxDims := 10 // 合理的最大维度数
	shape := make([]C.int, maxDims)
	C.MNN_Express_VARP_getShape(v.Handle, &shape[0], C.int(maxDims))

	// 计算实际维度数
	dimSize := 0
	for dimSize < maxDims {
		if shape[dimSize] == 0 {
			break
		}
		dimSize++
	}

	// 转换为Go切片
	goShape := make([]int, dimSize)
	for i := 0; i < dimSize; i++ {
		goShape[i] = int(shape[i])
	}

	return goShape
}

// ExpressModule 是MNN Express Module的Go封装
type ExpressModule struct {
	Handle  *C.MNN_Express_Module
	IsValid bool
}

// NewExpressModuleFromFile 从文件加载Module
func NewExpressModuleFromFile(
	inputs []string,
	outputs []string,
	fileName string,
	runtimeManager *ExpressRuntimeManager,
	config ExpressConfig,
) (*ExpressModule, error) {
	if len(inputs) == 0 || len(outputs) == 0 || fileName == "" {
		return nil, fmt.Errorf("inputs, outputs or fileName is empty")
	}

	cInputs := make([]*C.char, len(inputs))
	for i, input := range inputs {
		cInputs[i] = C.CString(input)
		defer C.free(unsafe.Pointer(cInputs[i]))
	}

	cOutputs := make([]*C.char, len(outputs))
	for i, output := range outputs {
		cOutputs[i] = C.CString(output)
		defer C.free(unsafe.Pointer(cOutputs[i]))
	}

	cFileName := C.CString(fileName)
	defer C.free(unsafe.Pointer(cFileName))

	var cRuntimeManager *C.MNN_Express_RuntimeManager
	if runtimeManager != nil && runtimeManager.IsValid {
		cRuntimeManager = runtimeManager.Handle
	}

	cConfig := config.ToCExpressConfig()

	module := C.MNN_Express_Module_loadFromFile(
		&cInputs[0],
		C.int(len(inputs)),
		&cOutputs[0],
		C.int(len(outputs)),
		cFileName,
		cRuntimeManager,
		&cConfig,
	)

	if module == nil {
		return nil, fmt.Errorf("failed to load module from file")
	}

	return &ExpressModule{Handle: module, IsValid: true}, nil
}

// Close 关闭Module
func (m *ExpressModule) Close() {
	if m.IsValid {
		C.MNN_Express_Module_destroy(m.Handle)
		m.IsValid = false
	}
}

// Forward 执行前向传播
func (m *ExpressModule) Forward(inputs []*ExpressVARP) ([]*ExpressVARP, error) {
	if !m.IsValid || len(inputs) == 0 {
		return nil, fmt.Errorf("module is invalid or inputs is empty")
	}

	// 检查所有输入是否有效
	for _, input := range inputs {
		if !input.IsValid {
			return nil, fmt.Errorf("input VARP is invalid")
		}
	}

	// 转换输入为C数组
	cInputs := make([]*C.MNN_Express_VARP, len(inputs))
	for i, input := range inputs {
		cInputs[i] = input.Handle
	}

	// 执行前向传播
	var outputCount C.int
	cOutputs := C.MNN_Express_Module_onForward(m.Handle, &cInputs[0], C.int(len(inputs)), &outputCount)
	if cOutputs == nil {
		return nil, fmt.Errorf("failed to forward")
	}

	// 转换输出为Go切片
	outputVars := make([]*ExpressVARP, int(outputCount))
	if outputCount > 0 {
		slice := unsafe.Slice(cOutputs, outputCount)
		for i := 0; i < int(outputCount); i++ {
			outputVars[i] = &ExpressVARP{
				Handle:  slice[i], //cOutputs[i],
				IsValid: true,
			}
		}
	}

	// 释放C数组内存
	C.MNN_Express_freeVARPArray(cOutputs, outputCount)

	return outputVars, nil
}

// ExpressRuntimeManager 是MNN Express RuntimeManager的Go封装
type ExpressRuntimeManager struct {
	Handle  *C.MNN_Express_RuntimeManager
	IsValid bool
}

// NewExpressRuntimeManager 创建RuntimeManager
func NewExpressRuntimeManager(type_ ForwardType, numThread int) (*ExpressRuntimeManager, error) {
	runtimeManager := C.MNN_Express_RuntimeManager_create(C.MNN_C_ForwardType(type_), C.int(numThread))
	if runtimeManager == nil {
		return nil, fmt.Errorf("failed to create runtime manager")
	}

	return &ExpressRuntimeManager{Handle: runtimeManager, IsValid: true}, nil
}

// Close 关闭RuntimeManager
func (rm *ExpressRuntimeManager) Close() {
	if rm.IsValid {
		C.MNN_Express_RuntimeManager_destroy(rm.Handle)
		rm.IsValid = false
	}
}

// SetHint 设置运行时提示
func (rm *ExpressRuntimeManager) SetHint(hint, value int) {
	if rm.IsValid {
		C.MNN_Express_RuntimeManager_setHint(rm.Handle, C.int(hint), C.int(value))
	}
}

// -------------------------- Express Math Operations --------------------------

// ExpressAdd 加法操作
func ExpressAdd(x, y *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid || !y.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_add(x.Handle, y.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to add")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressSubtract 减法操作
func ExpressSubtract(x, y *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid || !y.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_subtract(x.Handle, y.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to subtract")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressMultiply 乘法操作
func ExpressMultiply(x, y *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid || !y.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_multiply(x.Handle, y.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to multiply")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressDivide 除法操作
func ExpressDivide(x, y *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid || !y.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_divide(x.Handle, y.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to divide")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressPow 幂运算
func ExpressPow(x, y *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid || !y.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_pow(x.Handle, y.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to pow")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressMinimum 最小值操作
func ExpressMinimum(x, y *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid || !y.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_minimum(x.Handle, y.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to minimum")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressMaximum 最大值操作
func ExpressMaximum(x, y *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid || !y.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_maximum(x.Handle, y.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to maximum")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressBiasAdd 偏置加法
func ExpressBiasAdd(value, bias *ExpressVARP) (*ExpressVARP, error) {
	if !value.IsValid || !bias.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_biasAdd(value.Handle, bias.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to bias add")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressGreater 大于比较
func ExpressGreater(x, y *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid || !y.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_greater(x.Handle, y.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to greater")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressGreaterEqual 大于等于比较
func ExpressGreaterEqual(x, y *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid || !y.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_greaterEqual(x.Handle, y.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to greater equal")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressLess 小于比较
func ExpressLess(x, y *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid || !y.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_less(x.Handle, y.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to less")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressLessEqual 小于等于比较
func ExpressLessEqual(x, y *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid || !y.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_lessEqual(x.Handle, y.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to less equal")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressEqual 等于比较
func ExpressEqual(x, y *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid || !y.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_equal(x.Handle, y.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to equal")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressFloorDiv 向下取整除法
func ExpressFloorDiv(x, y *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid || !y.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_floorDiv(x.Handle, y.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to floor div")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressFloorMod 向下取整取模
func ExpressFloorMod(x, y *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid || !y.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_floorMod(x.Handle, y.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to floor mod")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressSquaredDifference 平方差
func ExpressSquaredDifference(x, y *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid || !y.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_squaredDifference(x.Handle, y.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to squared difference")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressAtan2 反正切2
func ExpressAtan2(x, y *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid || !y.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_atan2(x.Handle, y.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to atan2")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// -------------------------- Unary Operations --------------------------

// ExpressSign 符号函数
func ExpressSign(a *ExpressVARP) (*ExpressVARP, error) {
	if !a.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_sign(a.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to sign")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressAbs 绝对值
func ExpressAbs(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_abs(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to abs")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressNegative 取反
func ExpressNegative(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_negative(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to negative")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressFloor 向下取整
func ExpressFloor(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_floor(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to floor")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressRound 四舍五入
func ExpressRound(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_round(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to round")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressCeil 向上取整
func ExpressCeil(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_ceil(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to ceil")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressSquare 平方
func ExpressSquare(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_square(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to square")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressSqrt 平方根
func ExpressSqrt(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_sqrt(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to sqrt")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressRsqrt 平方根倒数
func ExpressRsqrt(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_rsqrt(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to rsqrt")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressExp 指数函数
func ExpressExp(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_exp(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to exp")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressLog 自然对数
func ExpressLog(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_log(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to log")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressSin 正弦函数
func ExpressSin(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_sin(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to sin")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressCos 余弦函数
func ExpressCos(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_cos(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to cos")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressTan 正切函数
func ExpressTan(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_tan(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to tan")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressAsin 反正弦函数
func ExpressAsin(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_asin(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to asin")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressAcos 反余弦函数
func ExpressAcos(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_acos(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to acos")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressAtan 反正切函数
func ExpressAtan(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_atan(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to atan")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressReciprocal 倒数
func ExpressReciprocal(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_reciprocal(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to reciprocal")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressTanh 双曲正切
func ExpressTanh(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_tanh(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to tanh")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressSigmoid Sigmoid函数
func ExpressSigmoid(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_sigmoid(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to sigmoid")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// -------------------------- Reduce Operations --------------------------

// ExpressReduceSum 求和
func ExpressReduceSum(input *ExpressVARP, axis []int, keepDims bool) (*ExpressVARP, error) {
	if !input.IsValid || len(axis) == 0 {
		return nil, fmt.Errorf("input VARP or axis is invalid")
	}

	cAxis := make([]C.int, len(axis))
	for i, a := range axis {
		cAxis[i] = C.int(a)
	}

	result := C.MNN_Express_reduceSum(input.Handle, &cAxis[0], C.int(len(axis)), toCBool(keepDims))
	if result == nil {
		return nil, fmt.Errorf("failed to reduce sum")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressReduceMean 求平均
func ExpressReduceMean(input *ExpressVARP, axis []int, keepDims bool) (*ExpressVARP, error) {
	if !input.IsValid || len(axis) == 0 {
		return nil, fmt.Errorf("input VARP or axis is invalid")
	}

	cAxis := make([]C.int, len(axis))
	for i, a := range axis {
		cAxis[i] = C.int(a)
	}

	result := C.MNN_Express_reduceMean(input.Handle, &cAxis[0], C.int(len(axis)), toCBool(keepDims))
	if result == nil {
		return nil, fmt.Errorf("failed to reduce mean")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressReduceMax 求最大值
func ExpressReduceMax(input *ExpressVARP, axis []int, keepDims bool) (*ExpressVARP, error) {
	if !input.IsValid || len(axis) == 0 {
		return nil, fmt.Errorf("input VARP or axis is invalid")
	}

	cAxis := make([]C.int, len(axis))
	for i, a := range axis {
		cAxis[i] = C.int(a)
	}

	result := C.MNN_Express_reduceMax(input.Handle, &cAxis[0], C.int(len(axis)), toCBool(keepDims))
	if result == nil {
		return nil, fmt.Errorf("failed to reduce max")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressReduceMin 求最小值
func ExpressReduceMin(input *ExpressVARP, axis []int, keepDims bool) (*ExpressVARP, error) {
	if !input.IsValid || len(axis) == 0 {
		return nil, fmt.Errorf("input VARP or axis is invalid")
	}

	cAxis := make([]C.int, len(axis))
	for i, a := range axis {
		cAxis[i] = C.int(a)
	}

	result := C.MNN_Express_reduceMin(input.Handle, &cAxis[0], C.int(len(axis)), toCBool(keepDims))
	if result == nil {
		return nil, fmt.Errorf("failed to reduce min")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressReduceProd 求乘积
func ExpressReduceProd(input *ExpressVARP, axis []int, keepDims bool) (*ExpressVARP, error) {
	if !input.IsValid || len(axis) == 0 {
		return nil, fmt.Errorf("input VARP or axis is invalid")
	}

	cAxis := make([]C.int, len(axis))
	for i, a := range axis {
		cAxis[i] = C.int(a)
	}

	result := C.MNN_Express_reduceProd(input.Handle, &cAxis[0], C.int(len(axis)), toCBool(keepDims))
	if result == nil {
		return nil, fmt.Errorf("failed to reduce prod")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// -------------------------- Other Operations --------------------------

// ExpressCastToFloat 转换为浮点型
func ExpressCastToFloat(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_castToFloat(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to cast to float")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressCastToInt 转换为整数型
func ExpressCastToInt(x *ExpressVARP) (*ExpressVARP, error) {
	if !x.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_castToInt(x.Handle)
	if result == nil {
		return nil, fmt.Errorf("failed to cast to int")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressMatMul 矩阵乘法
func ExpressMatMul(a, b *ExpressVARP, transposeA, transposeB bool) (*ExpressVARP, error) {
	if !a.IsValid || !b.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_matMul(a.Handle, b.Handle, toCBool(transposeA), toCBool(transposeB))
	if result == nil {
		return nil, fmt.Errorf("failed to matmul")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressArgMax 求最大值索引
func ExpressArgMax(input *ExpressVARP, axis int) (*ExpressVARP, error) {
	if !input.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_argMax(input.Handle, C.int(axis))
	if result == nil {
		return nil, fmt.Errorf("failed to argmax")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressArgMin 求最小值索引
func ExpressArgMin(input *ExpressVARP, axis int) (*ExpressVARP, error) {
	if !input.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_argMin(input.Handle, C.int(axis))
	if result == nil {
		return nil, fmt.Errorf("failed to argmin")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}

// ExpressBatchMatMul 批量矩阵乘法
func ExpressBatchMatMul(x, y *ExpressVARP, adjX, adjY bool) (*ExpressVARP, error) {
	if !x.IsValid || !y.IsValid {
		return nil, fmt.Errorf("input VARP is invalid")
	}

	result := C.MNN_Express_batchMatMul(x.Handle, y.Handle, toCBool(adjX), toCBool(adjY))
	if result == nil {
		return nil, fmt.Errorf("failed to batch matmul")
	}

	return &ExpressVARP{Handle: result, IsValid: true}, nil
}
