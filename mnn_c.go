package mnn

/*
#cgo CFLAGS: -I.
#cgo CXXFLAGS: -I.
#cgo linux LDFLAGS: -lMNN -lstdc++
#cgo windows LDFLAGS: libmnn.a -lstdc++
#include <MNN_C.h>
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// 新增转换工具函数
func toCBool(b bool) C.MNN_BOOL {
	if b {
		return C.MNN_BOOL(1)
	}
	return C.MNN_BOOL(0)
}

func toGoBool(b C.MNN_BOOL) bool {
	return b != 0
}

// ForwardType 定义MNN的前向推理类型
type ForwardType int

const (
	ForwardCPU          ForwardType = C.MNN_C_FORWARD_CPU
	ForwardMetal        ForwardType = C.MNN_C_FORWARD_METAL
	ForwardCUDA         ForwardType = C.MNN_C_FORWARD_CUDA
	ForwardOpenCL       ForwardType = C.MNN_C_FORWARD_OPENCL
	ForwardAuto         ForwardType = C.MNN_C_FORWARD_AUTO
	ForwardNN           ForwardType = C.MNN_C_FORWARD_NN
	ForwardOpenGL       ForwardType = C.MNN_C_FORWARD_OPENGL
	ForwardVulkan       ForwardType = C.MNN_C_FORWARD_VULKAN
	ForwardUser0        ForwardType = C.MNN_C_FORWARD_USER_0
	ForwardUser1        ForwardType = C.MNN_C_FORWARD_USER_1
	ForwardUser2        ForwardType = C.MNN_C_FORWARD_USER_2
	ForwardUser3        ForwardType = C.MNN_C_FORWARD_USER_3
	ForwardAll          ForwardType = C.MNN_C_FORWARD_ALL
	ForwardCPUExtension ForwardType = C.MNN_C_FORWARD_CPU_EXTENSION
)

// ErrorCode 定义MNN的错误码
type ErrorCode int

const (
	NoError      ErrorCode = C.NO_ERROR
	OutOfMemory  ErrorCode = C.OUT_OF_MEMORY
	NotSupport   ErrorCode = C.NOT_SUPPORT
	NoFile       ErrorCode = C.NO_FILE
	InvalidData  ErrorCode = C.INVALID_DATA
	UnknownError ErrorCode = C.UNKNOWN_ERROR
)

// DimensionType 定义MNN的维度类型
type DimensionType int

const (
	DimensionTensorflow DimensionType = C.MNN_TENSORFLOW
	DimensionCaffe      DimensionType = C.MNN_CAFFE
	DimensionCaffeC4    DimensionType = C.MNN_CAFFE_C4
)

// ExpressDimensionFormat 定义MNN Express的维度格式
type ExpressDimensionFormat int

const (
	ExpressNHWC   ExpressDimensionFormat = C.MNN_EXPRESS_NHWC
	ExpressNC4HW4 ExpressDimensionFormat = C.MNN_EXPRESS_NC4HW4
	ExpressNCHW   ExpressDimensionFormat = C.MNN_EXPRESS_NCHW
)

// ImageFormat 定义图像格式
type ImageFormat int

const (
	ImageRGBA    ImageFormat = C.MNN_CV_RGBA
	ImageRGB     ImageFormat = C.MNN_CV_RGB
	ImageBGR     ImageFormat = C.MNN_CV_BGR
	ImageGRAY    ImageFormat = C.MNN_CV_GRAY
	ImageBGRA    ImageFormat = C.MNN_CV_BGRA
	ImageYCrCb   ImageFormat = C.MNN_CV_YCrCb
	ImageYUV     ImageFormat = C.MNN_CV_YUV
	ImageHSV     ImageFormat = C.MNN_CV_HSV
	ImageXYZ     ImageFormat = C.MNN_CV_XYZ
	ImageBGR555  ImageFormat = C.MNN_CV_BGR555
	ImageBGR565  ImageFormat = C.MNN_CV_BGR565
	ImageYUVNV21 ImageFormat = C.MNN_CV_YUV_NV21
	ImageYUVNV12 ImageFormat = C.MNN_CV_YUV_NV12
	ImageYUVI420 ImageFormat = C.MNN_CV_YUV_I420
	ImageHSVFull ImageFormat = C.MNN_CV_HSV_FULL
)

// Filter 定义图像过滤器类型
type Filter int

const (
	FilterNearest  Filter = C.MNN_CV_NEAREST
	FilterBilinear Filter = C.MNN_CV_BILINEAR
	FilterBicubic  Filter = C.MNN_CV_BICUBIC
)

// Wrap 定义图像边界处理方式
type Wrap int

const (
	WrapClampToEdge Wrap = C.MNN_CV_CLAMP_TO_EDGE
	WrapZero        Wrap = C.MNN_CV_ZERO
	WrapRepeat      Wrap = C.MNN_CV_REPEAT
)

// ScheduleConfig 定义MNN的调度配置
type ScheduleConfig struct {
	Type       ForwardType
	NumThread  int
	BackupType ForwardType
}

// ToCScheduleConfig 将Go的ScheduleConfig转换为C的MNN_ScheduleConfig
func (sc *ScheduleConfig) ToCScheduleConfig() C.MNN_ScheduleConfig {
	return C.MNN_ScheduleConfig{
		datatype:   C.MNN_C_ForwardType(sc.Type),
		numThread:  C.int(sc.NumThread),
		backupType: C.MNN_C_ForwardType(sc.BackupType),
	}
}

// ImageProcessConfig 定义图像处理配置
type ImageProcessConfig struct {
	FilterType   Filter
	SourceFormat ImageFormat
	DestFormat   ImageFormat
	Mean         [4]float32
	Normal       [4]float32
	Wrap         Wrap
}

// ToCImageProcessConfig 将Go的ImageProcessConfig转换为C的MNN_CV_ImageProcess_Config
func (ipc *ImageProcessConfig) ToCImageProcessConfig() C.MNN_CV_ImageProcess_Config {
	var mean [4]C.float
	var normal [4]C.float
	for i := 0; i < 4; i++ {
		mean[i] = C.float(ipc.Mean[i])
		normal[i] = C.float(ipc.Normal[i])
	}

	return C.MNN_CV_ImageProcess_Config{
		filterType:   C.MNN_CV_Filter(ipc.FilterType),
		sourceFormat: C.MNN_CV_ImageFormat(ipc.SourceFormat),
		destFormat:   C.MNN_CV_ImageFormat(ipc.DestFormat),
		mean:         mean,
		normal:       normal,
		wrap:         C.MNN_CV_Wrap(ipc.Wrap),
	}
}

// Interpreter 是MNN解释器的Go封装
type Interpreter struct {
	Handle  *C.MNN_Interpreter
	IsValid bool
}

// NewInterpreterFromFile 从文件创建Interpreter
func NewInterpreterFromFile(file string) (*Interpreter, error) {
	cFile := C.CString(file)
	defer C.free(unsafe.Pointer(cFile))

	handle := C.MNN_Interpreter_createFromFile(cFile)
	if handle == nil {
		return nil, fmt.Errorf("failed to create interpreter from file")
	}

	return &Interpreter{Handle: handle, IsValid: true}, nil
}

// NewInterpreterFromBuffer 从缓冲区创建Interpreter
func NewInterpreterFromBuffer(buffer []byte) (*Interpreter, error) {
	if len(buffer) == 0 {
		return nil, fmt.Errorf("buffer is empty")
	}

	handle := C.MNN_Interpreter_createFromBuffer(unsafe.Pointer(&buffer[0]), C.size_t(len(buffer)))
	if handle == nil {
		return nil, fmt.Errorf("failed to create interpreter from buffer")
	}

	return &Interpreter{Handle: handle, IsValid: true}, nil
}

// Close 关闭Interpreter
func (i *Interpreter) Close() {
	if i.IsValid {
		C.MNN_Interpreter_destroy(i.Handle)
		i.IsValid = false
	}
}

// CreateSession 创建会话
func (i *Interpreter) CreateSession(config ScheduleConfig) (*Session, error) {
	if !i.IsValid {
		return nil, fmt.Errorf("interpreter instance is invalid")
	}

	cConfig := config.ToCScheduleConfig()
	session := C.MNN_Interpreter_createSession(i.Handle, &cConfig)
	if session == nil {
		return nil, fmt.Errorf("failed to create session")
	}

	return &Session{Handle: session, IsValid: true, Interpreter: i}, nil
}

// ReleaseSession 释放会话
func (i *Interpreter) ReleaseSession(session *Session) bool {
	if !i.IsValid || !session.IsValid {
		return false
	}

	result := toGoBool(C.MNN_Interpreter_releaseSession(i.Handle, session.Handle))
	if result {
		session.IsValid = false
	}
	return result
}

// ResizeSession 调整会话大小
func (i *Interpreter) ResizeSession(session *Session) {
	if i.IsValid && session.IsValid {
		C.MNN_Interpreter_resizeSession(i.Handle, session.Handle)
	}
}

// ResizeTensor 调整张量大小
func (i *Interpreter) ResizeTensor(tensor *Tensor, dims []int) {
	if i.IsValid && tensor.IsValid {
		cDims := make([]C.int, len(dims))
		for j, d := range dims {
			cDims[j] = C.int(d)
		}
		C.MNN_Interpreter_resizeTensor(i.Handle, tensor.Handle, &cDims[0], C.int(len(dims)))
	}
}

// RunSession 运行会话
func (i *Interpreter) RunSession(session *Session) ErrorCode {
	if !i.IsValid || !session.IsValid {
		return InvalidData
	}

	return ErrorCode(C.MNN_Interpreter_runSession(i.Handle, session.Handle))
}

// Session 是MNN会话的Go封装
type Session struct {
	Handle      *C.MNN_Session
	IsValid     bool
	Interpreter *Interpreter
}

// GetInput 获取会话输入张量
func (s *Session) GetInput(name string) (*Tensor, error) {
	if !s.IsValid || !s.Interpreter.IsValid {
		return nil, fmt.Errorf("session or interpreter is invalid")
	}

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	tensor := C.MNN_Interpreter_getSessionInput(s.Interpreter.Handle, s.Handle, cName)
	if tensor == nil {
		return nil, fmt.Errorf("failed to get input tensor: %s", name)
	}

	return &Tensor{Handle: tensor, IsValid: true}, nil
}

// GetOutput 获取会话输出张量
func (s *Session) GetOutput(name string) (*Tensor, error) {
	if !s.IsValid || !s.Interpreter.IsValid {
		return nil, fmt.Errorf("session or interpreter is invalid")
	}

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	tensor := C.MNN_Interpreter_getSessionOutput(s.Interpreter.Handle, s.Handle, cName)
	if tensor == nil {
		return nil, fmt.Errorf("failed to get output tensor: %s", name)
	}

	return &Tensor{Handle: tensor, IsValid: true}, nil
}

// Tensor 是MNN张量的Go封装
type Tensor struct {
	Handle  *C.MNN_Tensor
	IsValid bool
}

// CreateHostTensorFromDevice 从设备张量创建主机张量
func CreateHostTensorFromDevice(deviceTensor *Tensor, copyData bool) (*Tensor, error) {
	if !deviceTensor.IsValid {
		return nil, fmt.Errorf("device tensor is invalid")
	}

	tensor := C.MNN_Tensor_createHostTensorFromDevice(deviceTensor.Handle, toCBool(copyData))
	if tensor == nil {
		return nil, fmt.Errorf("failed to create host tensor from device tensor")
	}

	return &Tensor{Handle: tensor, IsValid: true}, nil
}

// CreateTensor 创建张量
func CreateTensor(tensor *Tensor, dimType DimensionType, allocMemory bool) (*Tensor, error) {
	if !tensor.IsValid {
		return nil, fmt.Errorf("tensor is invalid")
	}

	newTensor := C.MNN_Tensor_create(tensor.Handle, C.MNN_DimensionType(dimType), toCBool(allocMemory))
	if newTensor == nil {
		return nil, fmt.Errorf("failed to create tensor")
	}

	return &Tensor{Handle: newTensor, IsValid: true}, nil
}

// Close 关闭张量
func (t *Tensor) Close() {
	if t.IsValid {
		C.MNN_Tensor_destroy(t.Handle)
		t.IsValid = false
	}
}

// CopyFromHostTensor 从主机张量复制数据到设备张量
func (t *Tensor) CopyFromHostTensor(hostTensor *Tensor) bool {
	if !t.IsValid || !hostTensor.IsValid {
		return false
	}

	return toGoBool(C.MNN_Tensor_copyFromHostTensor(t.Handle, hostTensor.Handle))
}

// CopyToHostTensor 从设备张量复制数据到主机张量
func (t *Tensor) CopyToHostTensor(hostTensor *Tensor) bool {
	if !t.IsValid || !hostTensor.IsValid {
		return false
	}

	return toGoBool(C.MNN_Tensor_copyToHostTensor(t.Handle, hostTensor.Handle))
}

// GetFloatData 获取浮点数据指针
func (t *Tensor) GetFloatData() []float32 {
	if !t.IsValid {
		return nil
	}

	data := C.MNN_Tensor_getFloatData(t.Handle)
	if data == nil {
		return nil
	}

	elementSize := int(C.MNN_Tensor_elementSize(t.Handle))
	return (*[1 << 30]float32)(unsafe.Pointer(data))[:elementSize]
}

// ElementSize 获取元素数量
func (t *Tensor) ElementSize() int {
	if !t.IsValid {
		return 0
	}

	return int(C.MNN_Tensor_elementSize(t.Handle))
}

// GetShape 获取张量形状
func (t *Tensor) GetShape() []int {
	if !t.IsValid {
		return nil
	}

	// 先获取元素数量来估算最大维度数
	elementSize := t.ElementSize()
	if elementSize == 0 {
		return nil
	}

	// 尝试获取形状
	maxDims := 10 // 合理的最大维度数
	shape := make([]C.int, maxDims)
	C.MNN_Tensor_getShape(t.Handle, &shape[0], C.int(maxDims))

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

// ImageProcess 是MNN图像处理的Go封装
type ImageProcess struct {
	Handle  *C.MNN_CV_ImageProcess
	IsValid bool
}

// CreateImageProcess 创建图像处理器
func CreateImageProcess(config ImageProcessConfig, dstTensor *Tensor) (*ImageProcess, error) {
	cConfig := config.ToCImageProcessConfig()

	imageProcess := C.MNN_CV_ImageProcess_create(&cConfig, dstTensor.Handle)
	if imageProcess == nil {
		return nil, fmt.Errorf("failed to create image process")
	}

	return &ImageProcess{Handle: imageProcess, IsValid: true}, nil
}

// Close 关闭图像处理器
func (ip *ImageProcess) Close() {
	if ip.IsValid {
		C.MNN_CV_ImageProcess_destroy(ip.Handle)
		ip.IsValid = false
	}
}

// SetMatrix 设置变换矩阵
func (ip *ImageProcess) SetMatrix(matrix [9]float32) {
	if ip.IsValid {
		var cMatrix [9]C.float
		for i, v := range matrix {
			cMatrix[i] = C.float(v)
		}
		C.MNN_CV_ImageProcess_setMatrix(ip.Handle, &cMatrix[0])
	}
}

// Convert 转换图像
func (ip *ImageProcess) Convert(source []byte, iw, ih, stride int, dest *Tensor) ErrorCode {
	if !ip.IsValid || !dest.IsValid {
		return InvalidData
	}

	return ErrorCode(C.MNN_CV_ImageProcess_convert(ip.Handle, (*C.uint8_t)(&source[0]), C.int(iw), C.int(ih), C.int(stride), dest.Handle))
}

// SetPadding 设置填充值
func (ip *ImageProcess) SetPadding(value uint8) {
	if ip.IsValid {
		C.MNN_CV_ImageProcess_setPadding(ip.Handle, C.uint8_t(value))
	}
}

// GetVersion 获取MNN版本
func GetVersion() string {
	return C.GoString(C.MNN_getVersion())
}
