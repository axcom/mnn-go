package main

import (
	"fmt"
	"image/jpeg"
	"math"
	"os"

	"github.com/axcom/mnn"
	"github.com/nfnt/resize"
)

// 定义常量
const (
	IMAGE_VERIFY_SIZE = 224
	CLASSES_SIZE      = 1000
)

func main() {
	// 定义常量
	const (
		INPUT_NAME  = "input"
		OUTPUT_NAME = "output"
	)

	if len(os.Args) < 3 {
		fmt.Printf("Usage:\n\t%s mnn_model_path image_path\n", os.Args[0])
		os.Exit(-1)
	}

	// 获取命令行参数
	mnnModelPath := os.Args[1]
	imagePath := os.Args[2]

	// 创建解释器
	mnnNet, err := mnn.NewInterpreterFromFile(mnnModelPath)
	if err != nil {
		fmt.Printf("Failed to create interpreter: %v\n", err)
		os.Exit(-1)
	}
	defer mnnNet.Close()

	// 配置会话
	config := mnn.ScheduleConfig{
		Type:       mnn.ForwardCPU,
		NumThread:  4,
		BackupType: mnn.ForwardCPU,
	}

	// 创建会话
	session, err := mnnNet.CreateSession(config)
	if err != nil {
		fmt.Printf("Failed to create session: %v\n", err)
		os.Exit(-1)
	}
	defer mnnNet.ReleaseSession(session)

	// 获取输入张量
	input, err := session.GetInput(INPUT_NAME)
	if err != nil {
		fmt.Printf("Failed to get session input: %v\n", err)
		os.Exit(-1)
	}

	// 调整输入大小
	if input.ElementSize() <= 4 {
		dims := []int{1, 3, IMAGE_VERIFY_SIZE, IMAGE_VERIFY_SIZE}
		mnnNet.ResizeTensor(input, dims)
		mnnNet.ResizeSession(session)
	}

	// 获取输入形状
	inputShape := input.GetShape()
	fmt.Printf("input shape: %v\n", inputShape)

	// 创建主机张量
	givenTensor, err := mnn.CreateTensor(input, mnn.DimensionCaffe, true)
	if err != nil {
		fmt.Printf("Failed to create given tensor: %v\n", err)
		os.Exit(-1)
	}
	defer givenTensor.Close()

	// 获取输入数据指针
	inputData := givenTensor.GetFloatData()
	if inputData == nil {
		fmt.Println("Failed to get tensor data")
		os.Exit(-1)
	}

	// 预处理图像
	// 加载图像
	file, err := os.Open(imagePath)
	if err != nil {
		fmt.Printf("Failed to load image: %v\n", err)
		os.Exit(-1)
	}
	defer file.Close()

	// 解码JPEG图像
	img, err := jpeg.Decode(file)
	if err != nil {
		fmt.Printf("Failed to decode image: %v\n", err)
		os.Exit(-1)
	}

	// 调整图像大小
	resizedImg := resize.Resize(uint(IMAGE_VERIFY_SIZE), uint(IMAGE_VERIFY_SIZE), img, resize.Lanczos3)

	// 归一化图像数据
	for k := 0; k < 3; k++ {
		for i := 0; i < IMAGE_VERIFY_SIZE; i++ {
			for j := 0; j < IMAGE_VERIFY_SIZE; j++ {
				r, g, b, _ := resizedImg.At(j, i).RGBA()

				var src float32
				if k == 0 {
					src = float32(b >> 8)
				} else if k == 1 {
					src = float32(g >> 8)
				} else {
					src = float32(r >> 8)
				}

				var dst float32
				if k == 0 {
					dst = (src / 255.0) - 0.485
					dst /= 0.229
				} else if k == 1 {
					dst = (src / 255.0) - 0.456
					dst /= 0.224
				} else {
					dst = (src / 255.0) - 0.406
					dst /= 0.225
				}

				// 直接操作Go切片
				inputData[k*IMAGE_VERIFY_SIZE*IMAGE_VERIFY_SIZE+i*IMAGE_VERIFY_SIZE+j] = dst
			}
		}
	}

	// 复制数据到设备张量
	if !input.CopyFromHostTensor(givenTensor) {
		fmt.Println("Failed to copy from host tensor")
		os.Exit(-1)
	}

	// 运行会话
	if mnnNet.RunSession(session) != mnn.NoError {
		fmt.Println("Failed to run session")
		os.Exit(-1)
	}

	// 获取输出张量
	output, err := session.GetOutput(OUTPUT_NAME)
	if err != nil {
		fmt.Printf("Failed to get session output: %v\n", err)
		os.Exit(-1)
	}

	// 创建主机输出张量
	outputHost, err := mnn.CreateHostTensorFromDevice(output, true)
	if err != nil {
		fmt.Printf("Failed to create host tensor from device: %v\n", err)
		os.Exit(-1)
	}
	defer outputHost.Close()

	// 复制数据到主机张量
	if !output.CopyToHostTensor(outputHost) {
		fmt.Println("Failed to copy to host tensor")
		os.Exit(-1)
	}

	// 获取输出数据
	outputValues := outputHost.GetFloatData()
	if outputValues == nil {
		fmt.Println("Failed to get output values")
		os.Exit(-1)
	}

	// 后处理
	maxIndex := 0
	expSum := 0.0
	for i := 0; i < CLASSES_SIZE; i++ {
		val := outputValues[i]
		maxVal := outputValues[maxIndex]
		if val > maxVal {
			maxIndex = i
		}
		expSum += math.Exp(float64(val))
	}

	fmt.Printf("cls id: %d\n", maxIndex)
	fmt.Printf("cls prob: %f\n", math.Exp(float64(outputValues[maxIndex]))/expSum)
}
