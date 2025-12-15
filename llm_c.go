package mnn

/*
#cgo CFLAGS: -I.
#cgo CXXFLAGS: -I.
#cgo linux LDFLAGS: -lMNN -lMNN_Express -lllm -lstdc++
#cgo windows LDFLAGS: libmnn.a -lstdc++
#include <llm_c.h>
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Llm 是LLM的Go语言封装
type Llm struct {
	Handle  C.LlmHandle
	IsValid bool
}

// NewLlm 创建一个新的LLM实例
func NewLlm(configPath string) (*Llm, error) {
	cConfigPath := C.CString(configPath)
	defer C.free(unsafe.Pointer(cConfigPath))

	handle := C.LLM_Create(cConfigPath)
	// 检查 handle 是否有效
	// 我们需要一种方式来检查创建是否成功
	// 暂时假设成功创建

	return &Llm{Handle: handle, IsValid: true}, nil
}

// Close 关闭LLM实例
func (l *Llm) Close() {
	if l.IsValid {
		C.LLM_Destroy(l.Handle)
		l.IsValid = false
	}
}

// SetConfig 设置LLM配置
func (l *Llm) SetConfig(config string) error {
	if !l.IsValid {
		return fmt.Errorf("llm instance is invalid")
	}
	cConfig := C.CString(config)
	defer C.free(unsafe.Pointer(cConfig))

	if !bool(C.LLM_SetConfig(l.Handle, cConfig)) {
		return fmt.Errorf("failed to set config")
	}

	return nil
}

// Load 加载LLM模型
func (l *Llm) Load() error {
	if !l.IsValid {
		return fmt.Errorf("llm instance is invalid")
	}
	if !bool(C.LLM_Load(l.Handle)) {
		return fmt.Errorf("failed to load LLM model")
	}

	return nil
}

// Generate 生成文本
func (l *Llm) Generate(maxTokens int) (string, error) {
	if !l.IsValid {
		return "", fmt.Errorf("llm instance is invalid")
	}
	cResult := C.LLM_Generate(l.Handle, C.int(maxTokens))
	if cResult == nil {
		return "", fmt.Errorf("failed to generate text")
	}
	// 注意：这里的内存释放应该由C端负责，或者提供一个释放函数

	return C.GoString(cResult), nil
}

// Response 获取响应
func (l *Llm) Response(prompt string) (string, error) {
	if !l.IsValid {
		return "", fmt.Errorf("llm instance is invalid")
	}
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	cResult := C.LLM_Response(l.Handle, cPrompt, C.bool(false))
	if cResult == nil {
		return "", fmt.Errorf("failed to get response")
	}

	return C.GoString(cResult), nil
}

// Reset 重置LLM状态
func (l *Llm) Reset() error {
	if !l.IsValid {
		return fmt.Errorf("llm instance is invalid")
	}
	C.LLM_Reset(l.Handle)
	return nil
}

// IsStoped 检查生成是否已停止
func (l *Llm) IsStoped() bool {
	if !l.IsValid {
		return true
	}
	return bool(C.LLM_IsStoped(l.Handle))
}

// Note: Tuning functions are not available in the current LLM implementation

// Forward 处理输入
func (l *Llm) Forward(input string, imagePaths []string, audioPaths []string) (string, error) {
	if !l.IsValid {
		return "", fmt.Errorf("llm instance is invalid")
	}
	cInput := C.CString(input)
	defer C.free(unsafe.Pointer(cInput))

	// Convert image paths to C array
	var cImagePaths *C.char
	if len(imagePaths) > 0 {
		cImagePaths = C.CString(imagePaths[0]) // 简单处理，只支持单个图片
		defer C.free(unsafe.Pointer(cImagePaths))
	}

	// Convert audio paths to C array
	var cAudioPaths *C.char
	if len(audioPaths) > 0 {
		cAudioPaths = C.CString(audioPaths[0]) // 简单处理，只支持单个音频
		defer C.free(unsafe.Pointer(cAudioPaths))
	}

	cResult := C.LLM_Forward(l.Handle, cInput, cImagePaths, cAudioPaths)
	if cResult == nil {
		return "", fmt.Errorf("failed to forward input")
	}

	return C.GoString(cResult), nil
}

// LLMContext wraps the C LLM_Context struct
type LLMContext struct {
	C *C.LLM_Context
}

// GetContext returns the current LLM context
func (l *Llm) GetContext() *LLMContext {
	if !l.IsValid {
		return nil
	}
	return &LLMContext{
		C: C.LLM_GetContext(l.Handle),
	}
}
