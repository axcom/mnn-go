package mnn

/*
#include "MNN/MNNForwardType.h"
#include "Interpreter_c.h"
*/
import "C"

// ForwardType 定义MNN的前向推理类型
type MNNForwardType int

const (
	MNN_FORWARD_CPU           MNNForwardType = C.MNN_FORWARD_CPU
	MNN_FORWARD_METAL         MNNForwardType = C.MNN_FORWARD_METAL
	MNN_FORWARD_CUDA          MNNForwardType = C.MNN_FORWARD_CUDA
	MNN_FORWARD_OPENCL        MNNForwardType = C.MNN_FORWARD_OPENCL
	MNN_FORWARD_AUTO          MNNForwardType = C.MNN_FORWARD_AUTO
	MNN_FORWARD_NN            MNNForwardType = C.MNN_FORWARD_NN
	MNN_FORWARD_OPENGL        MNNForwardType = C.MNN_FORWARD_OPENGL
	MNN_FORWARD_VULKAN        MNNForwardType = C.MNN_FORWARD_VULKAN
	MNN_FORWARD_USER_0        MNNForwardType = C.MNN_FORWARD_USER_0
	MNN_FORWARD_USER_1        MNNForwardType = C.MNN_FORWARD_USER_1
	MNN_FORWARD_USER_2        MNNForwardType = C.MNN_FORWARD_USER_2
	MNN_FORWARD_USER_3        MNNForwardType = C.MNN_FORWARD_USER_3
	MNN_FORWARD_ALL           MNNForwardType = C.MNN_FORWARD_ALL
	MNN_FORWARD_CPU_EXTENSION MNNForwardType = C.MNN_FORWARD_CPU_EXTENSION

	MNN_MEMORY_AHARDWAREBUFFER MNNForwardType = C.MNN_MEMORY_AHARDWAREBUFFER

	MNN_CONVERT_QNN        MNNForwardType = C.MNN_CONVERT_QNN
	MNN_CONVERT_NEUROPILOT MNNForwardType = C.MNN_CONVERT_NEUROPILOT
	MNN_CONVERT_COREML     MNNForwardType = C.MNN_CONVERT_COREML
)

// MNNGpuMode GPU运行模式
type MNNGpuMode uint32

const (
	MNN_GPU_TUNING_NONE   MNNGpuMode = 1 << 0 // 禁用调优（OpenCL/Vulkan）
	MNN_GPU_TUNING_HEAVY  MNNGpuMode = 1 << 1 // 深度调优（OpenCL/Vulkan）
	MNN_GPU_TUNING_WIDE   MNNGpuMode = 1 << 2 // 广度调优（默认，OpenCL/Vulkan）
	MNN_GPU_TUNING_NORMAL MNNGpuMode = 1 << 3 // 常规调优（仅OpenCL）
	MNN_GPU_TUNING_FAST   MNNGpuMode = 1 << 4 // 快速调优（仅OpenCL）

	MNN_GPU_MEMORY_BUFFER MNNGpuMode = 1 << 6 // OpenCL缓冲区内存
	MNN_GPU_MEMORY_IMAGE  MNNGpuMode = 1 << 7 // OpenCL图像内存

	MNN_GPU_RECORD_OP    MNNGpuMode = 1 << 8 // 单OP单记录（仅OpenCL）
	MNN_GPU_RECORD_BATCH MNNGpuMode = 1 << 9 // 批量记录（OpenCL：10OP；Vulkan：共享命令缓冲区）
)

// MNNMemoryMode 内存模式
type MemoryMode int

const (
	Memory_Normal MemoryMode = 0 // 普通内存模式
	Memory_High   MemoryMode = 1 // 高内存（优先性能）
	Memory_Low    MemoryMode = 2 // 低内存（优先内存）
)

// MNNPowerMode 功耗模式
type PowerMode int

const (
	Power_Normal PowerMode = 0 // 普通功耗
	Power_High   PowerMode = 1 // 高性能（高功耗）
	Power_Low    PowerMode = 2 // 低功耗（低性能）
)

// MNNPrecisionMode 精度模式
type PrecisionMode int

const (
	Precision_Normal   PrecisionMode = 0 // 普通精度
	Precision_High     PrecisionMode = 1 // 高精度
	Precision_Low      PrecisionMode = 2 // 低精度
	Precision_Low_BF16 PrecisionMode = 3 // 低精度BF16
)

// MNNRuntimeStatus 运行时状态
type RuntimeStatus int

const (
	STATUS_SUPPORT_FP16        RuntimeStatus = 0 // 是否支持FP16
	STATUS_SUPPORT_DOT_PRODUCT RuntimeStatus = 1 // 是否支持点积
	STATUS_SUPPORT_POWER_LOW   RuntimeStatus = 2 // 是否支持低功耗
	STATUS_COUNT               RuntimeStatus = 3 // 状态总数
)

// ===================== 后端配置结构体 =====================
// MNNBackendConfig MNN后端配置
type BackendConfig struct {
	Memory    MemoryMode    // 内存模式
	Power     PowerMode     // 功耗模式
	Precision PrecisionMode // 精度模式

	SharedContext uintptr // 用户自定义共享上下文（替代C的union）
	Flags         int     // CPU后端标志位（替代C的union）
}

// NewBackendConfig 创建默认配置的后端配置实例
func NewBackendConfig() *BackendConfig {
	return &BackendConfig{
		Memory:    Memory_Normal,
		Power:     Power_Normal,
		Precision: Precision_Normal,
	}
}

func (bc *BackendConfig) ToCBackendConfig() C.MNN_BackendConfig {
	backendConfig := C.MNN_BackendConfig{
		memory:    C.MNN_MemoryMode(bc.Memory),
		power:     C.MNN_Gpu_PowerMode(bc.Power),
		precision: C.MNN_Gpu_PrecisionMode(bc.Precision),
	}
	if bc.Flags != 0 {
		backendConfig.flags = C.size_t(bc.Flags)
	} else {
		//backendConfig.sharedContext = bc.SharedContext
	}
	return backendConfig
}

// ===================== 工具函数 =====================

// CheckGpuModeValid 检查GPU模式是否与后端类型匹配
func CheckGpuModeValid(forwardType MNNForwardType, gpuMode MNNGpuMode) bool {
	// 非GPU后端，模式均有效
	if forwardType != MNN_FORWARD_OPENCL && forwardType != MNN_FORWARD_VULKAN {
		return true
	}

	// Vulkan仅支持NONE/HEAVY/WIDE调优 + BATCH记录
	if forwardType == MNN_FORWARD_VULKAN {
		invalid := gpuMode &^ (MNN_GPU_TUNING_NONE | MNN_GPU_TUNING_HEAVY | MNN_GPU_TUNING_WIDE | MNN_GPU_RECORD_BATCH)
		return invalid == 0
	}

	// OpenCL支持所有模式，仅检查掩码范围
	return gpuMode&^((1<<10)-1) == 0
}

// RuntimeStatusDesc 获取运行时状态的描述字符串
func RuntimeStatusDesc(status RuntimeStatus) string {
	switch status {
	case STATUS_SUPPORT_FP16:
		return "SUPPORT_FP16"
	case STATUS_SUPPORT_DOT_PRODUCT:
		return "SUPPORT_DOT_PRODUCT"
	case STATUS_SUPPORT_POWER_LOW:
		return "SUPPORT_POWER_LOW"
	default:
		return "UNKNOWN_STATUS"
	}
}
