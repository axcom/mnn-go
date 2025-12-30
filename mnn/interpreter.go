//
//  interpreter.go
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

package mnn

/*
#include <stdlib.h>
#include "Interpreter_c.h"
#include "MNN/MNNForwardType.h"

////////////////////////////////////////////////////////////////////////////////
// C全局变量：存储Go回调和用户数据（CGO回调中转）
static MNN_TensorCallBack go_basic_before = NULL;
static MNN_TensorCallBack go_basic_after = NULL;
static void* go_user_data = NULL;
// 基础回调桥接：Before
extern bool basic_before_bridge(const MNN_Tensor** tensors, size_t tensorCount, const char* opName, void* userData) {
    if (go_basic_before) {
        return go_basic_before(tensors, tensorCount, opName, userData);
    }
    return true;
}
// 基础回调桥接：After
extern bool basic_after_bridge(const MNN_Tensor** tensors, size_t tensorCount, const char* opName, void* userData) {
    if (go_basic_after) {
        return go_basic_after(tensors, tensorCount, opName, userData);
    }
    return true;
}
extern void set_go_callbacks(
    MNN_TensorCallBack basic_before, MNN_TensorCallBack basic_after,
    void* userData
) {
    go_basic_before = basic_before;
    go_basic_after = basic_after;
    go_user_data = userData;
}
// 声明Go函数的原型（与Go的函数签名一致）

////////////////////////////////////////////////////////////////////////////////
// C全局变量：存储Go回调和用户数据（CGO回调中转）
static MNN_TensorCallBackWithInfo go_info_before = NULL;
static MNN_TensorCallBackWithInfo go_info_after = NULL;
//static void* go_user_data = NULL;
// 扩展回调桥接：Before
bool info_before_bridge(const MNN_Tensor** tensors, size_t tensorCount, const MNN_OperatorInfo* info, void* userData) {
    if (go_info_before) {
        return go_info_before(tensors, tensorCount, info, userData);
    }
    return true;
}
// 扩展回调桥接：After
bool info_after_bridge(const MNN_Tensor** tensors, size_t tensorCount, const MNN_OperatorInfo* info, void* userData) {
    if (go_info_after) {
        return go_info_after(tensors, tensorCount, info, userData);
    }
    return true;
}
extern void set_go_info_callbacks(
    MNN_TensorCallBackWithInfo before, MNN_TensorCallBackWithInfo after,
    void* userData
) {
    go_info_before = before;
    go_info_after = after;
//    go_user_data = userData;
}
// 声明Go函数的原型（与Go的函数签名一致）
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// Interpreter represents a MNN interpreter
type Interpreter struct {
	c *C.struct_MNN_Interpreter
}

// Session represents a MNN session
type Session struct {
	c           *C.struct_MNN_Session
	Interpreter *Interpreter
}

// Backend represents a MNN backend
type Backend struct {
	c *C.struct_MNN_Backend
}

// OperatorInfo 算子信息（对应C的MNN_OperatorInfo）
type OperatorInfo struct {
	c *C.MNN_OperatorInfo // C侧算子信息句柄
}

// ScheduleConfig represents session schedule configuration
type ScheduleConfig struct {
	SaveTensors   []string
	Type          MNNForwardType
	NumThread     int
	Path          ScheduleConfigPath
	BackupType    MNNForwardType
	BackendConfig *BackendConfig //unsafe.Pointer

	pinner        runtime.Pinner // 使用 Pinner 固定指针
	c_saveTensors *C.StringArray
}

// ScheduleConfigPath represents subpath configuration
type ScheduleConfigPath struct {
	Inputs  []string
	Outputs []string
	Mode    PathMode

	c_Inputs  *C.StringArray
	c_Outputs *C.StringArray
}

// PathMode represents path running mode
type PathMode int

const (
	// PathModeOp represents Op mode
	PathModeOp PathMode = iota
	// PathModeTensor represents Tensor mode
	PathModeTensor
)

func NewScheduleConfig() *ScheduleConfig {
	cfg := &ScheduleConfig{}
	runtime.SetFinalizer(cfg, func(cfg *ScheduleConfig) {
		if cfg.c_saveTensors != nil {
			FreeCStringArray(cfg.c_saveTensors)
		}
		if cfg.Path.c_Inputs != nil {
			FreeCStringArray(cfg.Path.c_Inputs)
		}
		if cfg.Path.c_Outputs != nil {
			FreeCStringArray(cfg.Path.c_Outputs)
		}
	})
	return cfg
}

func (scp *ScheduleConfigPath) ToCScheduleConfigPath() C.MNN_ScheduleConfig_Path {
	path := C.MNN_ScheduleConfig_Path{
		inputs:  ToCStringArray(scp.Inputs),
		outputs: ToCStringArray(scp.Outputs),
		mode:    C.MNN_Path_Mode(scp.Mode),
	}

	if scp.c_Inputs != nil {
		FreeCStringArray(scp.c_Inputs)
	}
	scp.c_Inputs = &path.inputs
	if scp.c_Outputs != nil {
		FreeCStringArray(scp.c_Outputs)
	}
	scp.c_Outputs = &path.outputs
	return path
}

// ToCScheduleConfig 将Go的ScheduleConfig转换为C的MNN_ScheduleConfig
func (sc *ScheduleConfig) ToCScheduleConfig() C.MNN_ScheduleConfig {
	config := C.MNN_ScheduleConfig{
		saveTensors: ToCStringArray(sc.SaveTensors),
		type_:       C.MNNForwardType(sc.Type),
		numThread:   C.int(sc.NumThread),
		backupType:  C.MNNForwardType(sc.BackupType),
	}
	config.path = sc.Path.ToCScheduleConfigPath()
	if sc.BackendConfig != nil {
		backendConfig := sc.BackendConfig.ToCBackendConfig()
		sc.pinner.Pin(&backendConfig) // 固定嵌套指针对象
		config.backendConfig = &backendConfig
	}

	if sc.c_saveTensors != nil {
		FreeCStringArray(sc.c_saveTensors)
	}
	sc.c_saveTensors = &config.saveTensors
	return config
}

func (sc *ScheduleConfig) Unpin() {
	sc.pinner.Unpin() // 调用后立即解除固定
}

// SessionMode represents session mode
type SessionMode int

const (
	// SessionModeDebug represents debug mode
	SessionModeDebug SessionMode = iota
	// SessionModeRelease represents release mode
	SessionModeRelease
	// SessionModeInputInside represents input inside mode
	SessionModeInputInside
	// SessionModeInputUser represents input user mode
	SessionModeInputUser
	// SessionModeOutputInside represents output inside mode
	SessionModeOutputInside
	// SessionModeOutputUser represents output user mode
	SessionModeOutputUser
	// SessionModeResizeDirect represents resize direct mode
	SessionModeResizeDirect
	// SessionModeResizeDefer represents resize defer mode
	SessionModeResizeDefer
	// SessionModeBackendFix represents backend fix mode
	SessionModeBackendFix
	// SessionModeBackendAuto represents backend auto mode
	SessionModeBackendAuto
	// SessionModeMemoryCollect represents memory collect mode
	SessionModeMemoryCollect
	// SessionModeMemoryCache represents memory cache mode
	SessionModeMemoryCache
	// SessionModeCodegenDisable represents codegen disable mode
	SessionModeCodegenDisable
	// SessionModeCodegenEnable represents codegen enable mode
	SessionModeCodegenEnable
	// SessionModeResizeCheck represents resize check mode
	SessionModeResizeCheck
	// SessionModeResizeFix represents resize fix mode
	SessionModeResizeFix
	// SessionModeModuleForwardSeparate represents module forward separate mode
	SessionModeModuleForwardSeparate
	// SessionModeModuleForwardCombine represents module forward combine mode
	SessionModeModuleForwardCombine
)

// HintMode represents hint mode
type HintMode int

const (
	// HintModeMaxTuningNumber represents max tuning number
	HintModeMaxTuningNumber HintMode = iota
	// HintModeStrictCheckModel represents strict check model
	HintModeStrictCheckModel
	// HintModeMemAllocatorType represents mem allocator type
	HintModeMemAllocatorType
	// HintModeWinogradMemoryLevel represents winograd memory level
	HintModeWinogradMemoryLevel
	// HintModeGeometryComputeMask represents geometry compute mask
	HintModeGeometryComputeMask
	// HintModeDynamicQuantOptions represents dynamic quant options
	HintModeDynamicQuantOptions
	// HintModeCPULittlecoreDecreaseRate represents CPU littlecore decrease rate
	HintModeCPULittlecoreDecreaseRate
	// HintModeQKVQuantOptions represents QKV quant options
	HintModeQKVQuantOptions
	// HintModeKVcacheSizeLimit represents kvcache size limit
	HintModeKVcacheSizeLimit
	// HintModeOpEncoderNumberForCommit represents op encoder number for commit
	HintModeOpEncoderNumberForCommit
	// HintModeKVcacheInfo represents kvcache info
	HintModeKVcacheInfo
	// HintModeMmapFileSize represents mmap file size
	HintModeMmapFileSize
	// HintModeUseCachedMmap represents use cached mmap
	HintModeUseCachedMmap
	// HintModeInitThreadNumber represents init thread number
	HintModeInitThreadNumber
	// HintModeCPUCoreIDs represents CPU core IDs
	HintModeCPUCoreIDs
	// HintModeCPUSME2Instructions represents CPU SME2 instructions
	HintModeCPUSME2Instructions
	// HintModeCPUEnableKleidiAI represents CPU enable KleidiAI
	HintModeCPUEnableKleidiAI
)

// ExternalPathType represents external path type
type ExternalPathType int

const (
	// ExternalPathTypeKVcacheDir represents kvcache directory
	ExternalPathTypeKVcacheDir ExternalPathType = iota
	// ExternalPathTypeFeaturemapDir represents featuremap directory
	ExternalPathTypeFeaturemapDir
	// ExternalPathTypeWeightDir represents weight directory
	ExternalPathTypeWeightDir
	// ExternalPathTypeNPUFileDir represents NPU file directory
	ExternalPathTypeNPUFileDir
)

// GeometryComputeMask constants
const (
	// GeometryComputeMaskFuseRegion represents fuse region
	GeometryComputeMaskFuseRegion = C.MNN_GEOMETRY_COMPUTE_MASK_FUSEREGION
	// GeometryComputeMaskFuseRegionMulti represents fuse region multi
	GeometryComputeMaskFuseRegionMulti = C.MNN_GEOMETRY_COMPUTE_MASK_FUSEREGION_MULTI
	// GeometryComputeMaskUseLoop represents use loop
	GeometryComputeMaskUseLoop = C.MNN_GEOMETRY_COMPUTE_MASK_USELOOP
	// GeometryComputeMaskOpenCache represents open cache
	GeometryComputeMaskOpenCache = C.MNN_GEOMETRY_COMPUTE_MASK_OPENCACHE
	// GeometryComputeMaskAll represents all masks
	GeometryComputeMaskAll = C.MNN_GEOMETRY_COMPUTE_MASK_ALL
)

// SessionInfoCode represents session info code
type SessionInfoCode int

const (
	// SessionInfoCodeMemory represents memory info
	SessionInfoCodeMemory SessionInfoCode = iota
	// SessionInfoCodeFlops represents flops info
	SessionInfoCodeFlops
	// SessionInfoCodeBackends represents backends info
	SessionInfoCodeBackends
	// SessionInfoCodeResizeStatus represents resize status info
	SessionInfoCodeResizeStatus
	// SessionInfoCodeThreadNumber represents thread number info
	SessionInfoCodeThreadNumber
	// SessionInfoCodeAll represents all info
	SessionInfoCodeAll
)

// GetVersion gets MNN version info
func GetVersion() string {
	return C.GoString(C.MNN_getVersion())
}

// CreateInterpreterFromFile creates interpreter from file
func CreateInterpreterFromFile(file string) *Interpreter {
	cFile := C.CString(file)
	defer C.free(unsafe.Pointer(cFile))

	cInterpreter := C.MNN_Interpreter_createFromFile(cFile)
	if cInterpreter == nil {
		return nil
	}

	return &Interpreter{c: cInterpreter}
}

// CreateInterpreterFromBuffer creates interpreter from buffer
func CreateInterpreterFromBuffer(buffer []byte) *Interpreter {
	if len(buffer) == 0 {
		return nil
	}

	cInterpreter := C.MNN_Interpreter_createFromBuffer(unsafe.Pointer(&buffer[0]), C.size_t(len(buffer)))
	if cInterpreter == nil {
		return nil
	}

	return &Interpreter{c: cInterpreter}
}

// Destroy destroys the interpreter
func (i *Interpreter) Close() {
	if i.c != nil {
		C.MNN_Interpreter_destroy(i.c)
		i.c = nil
	}
}

// SetSessionMode sets session mode
func (i *Interpreter) SetSessionMode(mode SessionMode) {
	C.MNN_Interpreter_setSessionMode(i.c, C.enum_MNN_SessionMode(mode))
}

// SetCacheFile sets cache file
func (i *Interpreter) SetCacheFile(cacheFile string, keySize int) {
	cCacheFile := C.CString(cacheFile)
	defer C.free(unsafe.Pointer(cCacheFile))

	C.MNN_Interpreter_setCacheFile(i.c, cCacheFile, C.size_t(keySize))
}

// SetExternalFile sets external file
func (i *Interpreter) SetExternalFile(file string, flag int) {
	cFile := C.CString(file)
	defer C.free(unsafe.Pointer(cFile))

	C.MNN_Interpreter_setExternalFile(i.c, cFile, C.size_t(flag))
}

// UpdateCacheFile updates cache file
func (i *Interpreter) UpdateCacheFile(session *Session, flag int) ErrorCode {
	return ErrorCode(C.MNN_Interpreter_updateCacheFile(i.c, session.c, C.int(flag)))
}

// SetSessionHint sets session hint
func (i *Interpreter) SetSessionHint(hint HintMode, value int) {
	C.MNN_Interpreter_setSessionHint(i.c, C.enum_MNN_HintMode(hint), C.int(value))
}

// SetSessionHintArray sets session hint with array
func (i *Interpreter) SetSessionHintArray(hint HintMode, value []int) {
	if len(value) == 0 {
		return
	}

	C.MNN_Interpreter_setSessionHintArray(i.c, C.enum_MNN_HintMode(hint), (*C.int)(unsafe.Pointer(&value[0])), C.size_t(len(value)))
}

// CreateSession creates session with schedule config
func (i *Interpreter) CreateSession(config *ScheduleConfig) *Session {
	cConfig := config.ToCScheduleConfig()
	defer config.Unpin()
	cSession := C.MNN_Interpreter_createSession(i.c, &cConfig)
	if cSession == nil {
		return nil
	}

	return &Session{c: cSession, Interpreter: i}
}

// ReleaseSession releases session
func (i *Interpreter) ReleaseSession(session *Session) bool {
	return B2Go(C.MNN_Interpreter_releaseSession(i.c, session.c))
}

// ResizeSession resizes session
func (i *Interpreter) ResizeSession(session *Session) {
	C.MNN_Interpreter_resizeSession(i.c, session.c)
}

// ResizeSessionEx resizes session with needRelloc parameter
func (i *Interpreter) ResizeSessionEx(session *Session, needRelloc int) {
	C.MNN_Interpreter_resizeSessionEx(i.c, session.c, C.int(needRelloc))
}

// ReleaseModel releases model
func (i *Interpreter) ReleaseModel() {
	C.MNN_Interpreter_releaseModel(i.c)
}

// GetModelBuffer gets model buffer
func (i *Interpreter) GetModelBuffer() ([]byte, error) {
	var buffer unsafe.Pointer //*C.void
	var size C.size_t

	C.MNN_Interpreter_getModelBuffer(i.c, &buffer, &size)

	if buffer == nil {
		return nil, nil
	}

	return C.GoBytes(buffer, C.int(size)), nil
}

// GetModelVersion gets model version
func (i *Interpreter) GetModelVersion() string {
	return C.GoString(C.MNN_Interpreter_getModelVersion(i.c))
}

// UpdateSessionToModel updates session's tensor to model's const op
func (i *Interpreter) UpdateSessionToModel(session *Session) ErrorCode {
	return ErrorCode(C.MNN_Interpreter_updateSessionToModel(i.c, session.c))
}

// RunSession runs session
func (i *Interpreter) RunSession(session *Session) ErrorCode {
	return ErrorCode(C.MNN_Interpreter_runSession(i.c, session.c))
}

// GetSessionInput gets session input tensor
func (i *Interpreter) GetSessionInput(session *Session, name string) *Tensor {
	var cName *C.char
	if name != "" {
		cName = C.CString(name)
		defer C.free(unsafe.Pointer(cName))
	}

	cTensor := C.MNN_Interpreter_getSessionInput(i.c, session.c, cName)
	if cTensor == nil {
		return nil
	}

	return &Tensor{c: cTensor}
}

// GetSessionOutput gets session output tensor
func (i *Interpreter) GetSessionOutput(session *Session, name string) *Tensor {
	var cName *C.char
	if name != "" {
		cName = C.CString(name)
		defer C.free(unsafe.Pointer(cName))
	}

	cTensor := C.MNN_Interpreter_getSessionOutput(i.c, session.c, cName)
	if cTensor == nil {
		return nil
	}

	return &Tensor{c: cTensor}
}

// GetSessionInfo gets session info
func (i *Interpreter) GetSessionInfo(session *Session, code SessionInfoCode, ptr unsafe.Pointer) bool {
	return B2Go(C.MNN_Interpreter_getSessionInfo(i.c, session.c, C.enum_MNN_SessionInfoCode(code), ptr))
}

// ResizeTensor resizes tensor
func (i *Interpreter) ResizeTensor(tensor *Tensor, dims []int) {
	if len(dims) == 0 {
		return
	}

	C.MNN_Interpreter_resizeTensor(i.c, tensor.c, (*C.int)(unsafe.Pointer(&dims[0])), C.int(len(dims)))
}

// ResizeTensor4D resizes tensor by nchw
func (i *Interpreter) ResizeTensor4D(tensor *Tensor, batch, channel, height, width int) {
	C.MNN_Interpreter_resizeTensor4D(i.c, tensor.c, C.int(batch), C.int(channel), C.int(height), C.int(width))
}

// GetBackend gets backend used to create tensor
func (i *Interpreter) GetBackend(session *Session, tensor *Tensor) *Backend {
	cBackend := C.MNN_Interpreter_getBackend(i.c, session.c, tensor.c)
	if cBackend == nil {
		return nil
	}

	return &Backend{c: cBackend}
}

// BizCode gets business code
func (i *Interpreter) BizCode() string {
	return C.GoString(C.MNN_Interpreter_bizCode(i.c))
}

// UUID gets model UUID
func (i *Interpreter) UUID() string {
	return C.GoString(C.MNN_Interpreter_uuid(i.c))
}

func (i *Interpreter) RunSessionWithCallBack(session *Session, before, after unsafe.Pointer, sync bool, userData unsafe.Pointer) ErrorCode {
	// 将unsafe.Pointer转换为mnn包的_Ctype_MNN_TensorCallBack
	C.set_go_callbacks((C.MNN_TensorCallBack)(before), (C.MNN_TensorCallBack)(after), nil)

	rcode := C.MNN_Interpreter_runSessionWithCallBack(i.c, session.c,
		C.MNN_TensorCallBack(C.basic_before_bridge),
		C.MNN_TensorCallBack(C.basic_after_bridge),
		B2C(sync), // 同步执行
		userData,
	)
	return ErrorCode(rcode)
}

func (i *Interpreter) RunSessionWithCallBackInfo(session *Session, before, after unsafe.Pointer, sync bool, userData unsafe.Pointer) ErrorCode {
	// 将unsafe.Pointer转换为mnn包的_Ctype_MNN_TensorCallBack
	C.set_go_info_callbacks((C.MNN_TensorCallBack)(before), (C.MNN_TensorCallBack)(after), nil)

	rcode := C.MNN_Interpreter_runSessionWithCallBackInfo(i.c, session.c,
		C.MNN_TensorCallBackWithInfo(C.info_before_bridge),
		C.MNN_TensorCallBackWithInfo(C.info_after_bridge),
		B2C(sync), // 同步执行
		userData,
	)
	return ErrorCode(rcode)
}

// NamedTensor 张量名称和张量的映射
type NamedTensor struct {
	Name   string
	Tensor *Tensor
}

// GetSessionInputAll 获取所有输入张量
func (i *Interpreter) GetSessionInputAll(s *Session) []NamedTensor {
	if i.c == nil || s.c == nil {
		return nil
	}
	cList := C.MNN_Interpreter_GetSessionInputAll(i.c, s.c)
	defer C.MNN_NamedTensorList_Free(cList) // 释放C侧内存

	tensors := make([]NamedTensor, 0, cList.count)
	for j := 0; j < int(cList.count); j++ {
		cTensor := (*C.MNN_NamedTensor)(unsafe.Pointer(uintptr(unsafe.Pointer(cList.tensors)) + uintptr(j)*unsafe.Sizeof(*cList.tensors)))
		name := C.GoString(cTensor.name)
		tensor := &Tensor{c: cTensor.tensor}
		tensors = append(tensors, NamedTensor{Name: name, Tensor: tensor})
	}
	return tensors
}

// GetSessionOutputAll 获取所有输出张量
func (i *Interpreter) GetSessionOutputAll(s *Session) []NamedTensor {
	if i.c == nil || s.c == nil {
		return nil
	}
	cList := C.MNN_Interpreter_GetSessionOutputAll(i.c, s.c)
	defer C.MNN_NamedTensorList_Free(cList) // 释放C侧内存

	tensors := make([]NamedTensor, 0, cList.count)
	for j := 0; j < int(cList.count); j++ {
		cTensor := (*C.MNN_NamedTensor)(unsafe.Pointer(uintptr(unsafe.Pointer(cList.tensors)) + uintptr(j)*unsafe.Sizeof(*cList.tensors)))
		name := C.GoString(cTensor.name)
		tensor := &Tensor{c: cTensor.tensor}
		tensors = append(tensors, NamedTensor{Name: name, Tensor: tensor})
	}
	return tensors
}

// -------------------------- OperatorInfo 方法 --------------------------

// NewOperatorInfoFromC 创建OperatorInfo（内部使用）
func NewOperatorInfoFromC(cInfo *C.MNN_OperatorInfo) *OperatorInfo {
	if cInfo == nil {
		return nil
	}
	info := &OperatorInfo{c: cInfo}
	/*runtime.SetFinalizer(info, func(info *OperatorInfo) {
		info.Destroy()
	})*/
	return info
}

// GetName 获取算子名称
func (oi *OperatorInfo) GetName() (string, error) {
	if oi == nil || oi.c == nil {
		return "", fmt.Errorf("算子信息未初始化")
	}
	cName := C.MNN_OperatorInfo_name(oi.c)
	if cName == nil {
		return "", fmt.Errorf("获取算子名称失败")
	}
	return C.GoString(cName), nil
}

// GetType 获取算子类型
func (oi *OperatorInfo) GetType() string {
	cType := C.MNN_OperatorInfo_type(oi.c)
	return C.GoString(cType)
}

// GetFlops 获取算子FLOPS（百万次）
func (oi *OperatorInfo) GetFlops() float32 {
	return float32(C.MNN_OperatorInfo_flops(oi.c))
}

// -------------------------- Session 方法 --------------------------

// ReleaseSession releases session
func (s *Session) Close() bool {
	return s.Interpreter.ReleaseSession(s)
}

// ResizeSession resizes session
func (s *Session) Resize() {
	s.Interpreter.ResizeSession(s)
}

// RunSession runs session
func (s *Session) Run() ErrorCode {
	return s.Interpreter.RunSession(s)
}
func (s *Session) RunWithCallBack(before, after unsafe.Pointer, sync bool, userData unsafe.Pointer) ErrorCode {
	return s.Interpreter.RunSessionWithCallBack(s, before, after, sync, userData)
}
func (s *Session) RunWithCallBackInfo(before, after unsafe.Pointer, sync bool, userData unsafe.Pointer) ErrorCode {
	return s.Interpreter.RunSessionWithCallBackInfo(s, before, after, sync, userData)
}

// GetInput 获取会话输入张量
func (s *Session) GetInput(name string) *Tensor {
	return s.Interpreter.GetSessionInput(s, name)
}

// GetSessionInputAll 获取所有输入张量
func (s *Session) GetInputAll() []NamedTensor {
	return s.Interpreter.GetSessionInputAll(s)
}

// GetOutput 获取会话输出张量
func (s *Session) GetOutput(name string) *Tensor {
	return s.Interpreter.GetSessionOutput(s, name)
}

// GetSessionOutputAll 获取所有输出张量
func (s *Session) GetOutputAll() []NamedTensor {
	return s.Interpreter.GetSessionOutputAll(s)
}

func (s *Session) GetInfo(code SessionInfoCode, ptr unsafe.Pointer) bool {
	return s.Interpreter.GetSessionInfo(s, code, ptr)
}
