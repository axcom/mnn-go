# MNN Go 绑定

这是 MNN (Mobile Neural Network) 框架的 Go 语言绑定，提供了对 MNN、llm 和 MNN_Express API 的 Golang 封装。

## 目录结构

```
.
├── MNN_C.cpp          # MNN_C 的 C++ 实现
├── MNN_C.h            # MNN_C 的 C API 头文件
├── MNN_Express.cpp    # MNN_Express 的 C++ 实现
├── MNN_Express.h      # MNN_Express 的 C API 头文件
├── llm_c.cpp          # LLM C++ 实现
├── llm_c.h            # LLM C API 头文件
├── llm.go             # LLM 功能的 Go 绑定（llm_demo参考示例）
├── mnn_c.go           # MNN_C 的 Go 绑定
├── mnn_express.go     # MNN_Express 的 Go 绑定
└── README.md          # 本说明文件
```

## 环境配置

### 1. 安装 MNN 库

首先需要下载并编译 MNN 库：

```bash
# 克隆 MNN 仓库
git clone https://github.com/alibaba/MNN.git
cd MNN

# 编译 MNN
mkdir build && cd build
cmake .. -DMNN_BUILD_LLM=true -DMNN_ARM82=OFF -DMNN_BUILD_LLM_OMNI=ON -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true
make -j4
```

### 2. 配置环境变量

确保 MNN 库的头文件和库文件可以被找到：

```bash
# 设置 MNN 根目录
export MNN_ROOT=~/mnn #这里替换成你的MNN路径

# 设置 CGO 环境变量
export CGO_CFLAGS="-I$MNN_ROOT/include"
export CGO_CXXFLAGS="-I$MNN_ROOT/include -I$MNN_ROOT/transformers/llm/engine/include"
export CGO_LDFLAGS="-L$MNN_ROOT/build -L$MNN_ROOT/build/express"
```

## 使用示例

### MNN_C 使用示例

```go
package main

import (
	"fmt"
	"log"
	"github.com/axcom/mnn"
)

func main() {
	// 加载模型
	interpreter, err := mnn.NewInterpreterFromFile("model.mnn")
	if err != nil {
		log.Fatalf("Failed to create interpreter: %v", err)
	}
	defer interpreter.Close()

	// 创建会话
	config := mnn.ScheduleConfig{
		Type:      mnn.ForwardCPU,
		NumThread: 4,
	}
	session, err := interpreter.CreateSession(config)
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}

	// 获取输入张量
	input, err := session.GetInput("input")
	if err != nil {
		log.Fatalf("Failed to get input: %v", err)
	}

	// 设置输入数据
	// ...

	// 运行模型
	err = interpreter.RunSession(session)
	if err != mnn.NoError {
		log.Fatalf("Failed to run session: %v", err)
	}

	// 获取输出张量
	output, err := session.GetOutput("output")
	if err != nil {
		log.Fatalf("Failed to get output: %v", err)
	}

	// 获取输出数据
	outputData := output.GetFloatData()
	fmt.Println("Output:", outputData)
}
```

## API 文档

### MNN_C

#### Interpreter
- `NewInterpreterFromFile(file string) (*Interpreter, error)`: 从文件创建解释器
- `NewInterpreterFromBuffer(buffer []byte) (*Interpreter, error)`: 从缓冲区创建解释器
- `Close()`: 关闭解释器
- `CreateSession(config ScheduleConfig) (*Session, error)`: 创建会话
- `ReleaseSession(session *Session) bool`: 释放会话
- `ResizeSession(session *Session)`: 调整会话大小
- `ResizeTensor(tensor *Tensor, dims []int)`: 调整张量大小
- `RunSession(session *Session) ErrorCode`: 运行会话

#### Session
- `GetInput(name string) (*Tensor, error)`: 获取输入张量
- `GetOutput(name string) (*Tensor, error)`: 获取输出张量

#### Tensor
- `CreateHostTensorFromDevice(deviceTensor *Tensor, copyData bool) (*Tensor, error)`: 从设备张量创建主机张量
- `CreateTensor(tensor *Tensor, dimType DimensionType, allocMemory bool) (*Tensor, error)`: 创建张量
- `Close()`: 关闭张量
- `CopyFromHostTensor(hostTensor *Tensor) bool`: 从主机张量复制数据到设备张量
- `CopyToHostTensor(hostTensor *Tensor) bool`: 从设备张量复制数据到主机张量
- `GetFloatData() []float32`: 获取浮点数据指针
- `ElementSize() int`: 获取元素数量
- `GetShape() []int`: 获取张量形状

#### ImageProcess
- `CreateImageProcess(config ImageProcessConfig, dstTensor *Tensor) (*ImageProcess, error)`: 创建图像处理器
- `Close()`: 关闭图像处理器
- `SetMatrix(matrix [9]float32)`: 设置变换矩阵
- `Convert(source []byte, iw, ih, stride int, dest *Tensor) ErrorCode`: 转换图像
- `SetPadding(value uint8)`: 设置填充值

### MNN_Express

#### VARP
- `NewExpressVARPFromConstFloat(data []float32, dims []int) (*ExpressVARP, error)`: 创建浮点常量 VARP
- `NewExpressVARPFromConstInt(data []int32, dims []int) (*ExpressVARP, error)`: 创建整数常量 VARP
- `Close()`: 关闭 VARP
- `GetFloatData() []float32`: 获取浮点数据
- `ElementSize() int`: 获取元素数量
- `GetShape() []int`: 获取形状

#### Module
- `NewExpressModuleFromFile(inputs []string, outputs []string, fileName string, runtimeManager *ExpressRuntimeManager, config ExpressConfig) (*ExpressModule, error)`: 从文件加载 Module
- `Close()`: 关闭 Module
- `Forward(inputs []*ExpressVARP) ([]*ExpressVARP, error)`: 执行前向传播

#### RuntimeManager
- `NewExpressRuntimeManager(type_ ForwardType, numThread int) (*ExpressRuntimeManager, error)`: 创建 RuntimeManager
- `Close()`: 关闭 RuntimeManager
- `SetHint(hint, value int)`: 设置运行时提示

#### 数学运算

支持各种数学运算，包括：
- 基本运算：Add, Subtract, Multiply, Divide
- 矩阵运算：MatMul, BatchMatMul
- 比较运算：Greater, Less, Equal
- 一元运算：Abs, Exp, Log, Sin, Cos
- 归约运算：ReduceSum, ReduceMean, ReduceMax

### LLM

#### Llm
- `NewLlm(configPath string) (*Llm, error)`: 创建一个新的LLM实例
- `Close()`: 关闭LLM实例
- `SetConfig(config string) error`: 设置LLM配置
- `Load() error`: 加载LLM模型
- `Generate(maxTokens int) (string, error)`: 生成文本
- `Response(prompt string) (string, error)`: 获取响应
- `Reset() error`: 重置LLM状态
- `IsStoped() bool`: 检查生成是否已停止
- `Forward(input string, imagePaths []string, audioPaths []string) (string, error)`: 处理输入
- `GetContext() *LLMContext`: 获取当前LLM上下文

#### LLMContext
- 封装了C的LLM_Context结构体，用于管理LLM上下文

## 注意事项

1. 所有的 C 资源都需要手动释放，使用 `Close()` 方法或 defer 语句确保资源被正确释放。
2. 在调用方法前，检查对象的 `IsValid` 字段确保对象有效。
3. 确保在编译前正确配置 MNN 库的路径。

## 许可证

MIT
