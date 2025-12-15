package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"
	. "github.com/axcom/mnn"
)

func tuningPrepare(llm *Llm) {
	fmt.Println("Prepare for tuning opt Begin")
	// 注意：tuning方法在Go封装中没有实现，需要根据实际情况添加
	fmt.Println("Prepare for tuning opt End")
}

func benchmark(llm *Llm, prompts []string, maxTokenNumber int) int {
	promptLen := 0
	decodeLen := 0
	prefillTime := int64(0)
	decodeTime := int64(0)
	sampleTime := int64(0)

	// 设置最大新token数
	if maxTokenNumber > 0 {
		if err := llm.SetConfig(`{"max_new_tokens":1}`); err != nil {
			fmt.Printf("Error setting config: %v\n", err)
		}
	}

	for i, prompt := range prompts {
		// 跳过以#开头的提示
		if strings.HasPrefix(prompt, "#") {
			continue
		}

		fmt.Printf("Processing prompt %d: %s\n", i+1, prompt)

		if maxTokenNumber >= 0 {
			// 生成响应
			response, err := llm.Response(prompt)
			if err != nil {
				fmt.Printf("Error generating response: %v\n", err)
				continue
			}
			fmt.Print(response)

			// 继续生成直到达到最大token数或停止
			for j := 0; j < maxTokenNumber && !llm.IsStoped(); j++ {
				generated, err := llm.Generate(1)
				if err != nil {
					fmt.Printf("Error in generate: %v\n", err)
					break
				}
				fmt.Print(generated)
			}
		} else {
			// 无限制生成
			response, err := llm.Response(prompt)
			if err != nil {
				fmt.Printf("Error generating response: %v\n", err)
				continue
			}
			fmt.Print(response)
		}

		// 这里需要获取上下文信息来计算统计数据
		// 由于Go封装中没有直接提供上下文的详细信息，可能需要扩展C API或调整实现
	}

	// 打印基准测试结果
	fmt.Println("\n#################################")
	fmt.Printf("prompt tokens num = %d\n", promptLen)
	fmt.Printf("decode tokens num = %d\n", decodeLen)
	fmt.Printf("prefill time = %.2f s\n", float64(prefillTime)/1e9)
	fmt.Printf("decode time = %.2f s\n", float64(decodeTime)/1e9)
	fmt.Printf("sample time = %.2f s\n", float64(sampleTime)/1e9)

	if prefillTime > 0 {
		fmt.Printf("prefill speed = %.2f tok/s\n", float64(promptLen)/(float64(prefillTime)/1e9))
	}
	if decodeTime > 0 {
		fmt.Printf("decode speed = %.2f tok/s\n", float64(decodeLen)/(float64(decodeTime)/1e9))
	}
	fmt.Println("##################################")

	return 0
}

func readPrompts(promptFile string) ([]string, error) {
	fmt.Printf("prompt file is %s\n", promptFile)

	file, err := os.Open(promptFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var prompts []string

	// 读取所有内容
	content, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}

	// 按行分割
	lines := strings.Split(string(content), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if strings.HasSuffix(line, "\r") {
			line = line[:len(line)-1]
		}
		prompts = append(prompts, line)
	}

	return prompts, nil
}

func eval(llm *Llm, promptFile string, maxTokenNumber int) int {
	prompts, err := readPrompts(promptFile)
	if err != nil {
		fmt.Printf("Error reading prompt file: %v\n", err)
		return 1
	}

	if len(prompts) == 0 {
		fmt.Println("No prompts found in file")
		return 1
	}

	// 检查是否是CEVAL格式
	if prompts[0] == "id,question,A,B,C,D,answer" {
		fmt.Println("CEVAL format detected, but not implemented in Go version")
		return 0
	}

	// 执行基准测试
	return benchmark(llm, prompts, maxTokenNumber)
}

func chat(llm *Llm) {
	fmt.Println("Starting chat session. Type /exit to quit, /reset to reset conversation.")

	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("\nUser: ")
		userInput, err := reader.ReadString('\n')
		if err != nil {
			fmt.Printf("Error reading input: %v\n", err)
			continue
		}

		userInput = strings.TrimSpace(userInput)

		// 检查命令
		if userInput == "/exit" {
			fmt.Println("Exiting chat session")
			return
		}

		if userInput == "/reset" {
			if err := llm.Reset(); err != nil {
				fmt.Printf("Error resetting LLM: %v\n", err)
			} else {
				fmt.Println("Conversation reset")
			}
			continue
		}

		// 生成响应
		fmt.Print("\nA: ")
		response, err := llm.Response(userInput)
		if err != nil {
			fmt.Printf("Error generating response: %v\n", err)
			continue
		}
		fmt.Print(response)
	}
}

func main() {
	// 解析命令行参数
	flag.Parse()
	args := flag.Args()

	if len(args) < 1 {
		fmt.Println("Usage: " + os.Args[0] + " config.json <prompt.txt>")
		return
	}

	configPath := args[0]
	fmt.Printf("config path is %s\n", configPath)

	// 创建LLM实例
	llm, err := NewLlm(configPath)
	if err != nil {
		fmt.Printf("Failed to create LLM instance: %v\n", err)
		return
	}
	defer llm.Close()

	// 设置临时路径
	if err := llm.SetConfig(`{"tmp_path":"tmp"}`); err != nil {
		fmt.Printf("Error setting tmp_path: %v\n", err)
	}

	// 加载模型
	startTime := time.Now()
	if err := llm.Load(); err != nil {
		fmt.Printf("LLM init error: %v\n", err)
		return
	}
	duration := time.Since(startTime)
	fmt.Printf("Model loaded in %v\n", duration)

	// 执行调优准备
	startTime = time.Now()
	tuningPrepare(llm)
	duration = time.Since(startTime)
	fmt.Printf("Tuning prepared in %v\n", duration)

	// 如果没有提供提示文件，启动聊天模式
	if len(args) < 2 {
		chat(llm)
		return
	}

	// 解析最大token数
	maxTokenNumber := -1
	if len(args) >= 3 {
		if num, err := strconv.Atoi(args[2]); err == nil {
			maxTokenNumber = num
		}
	}

	// 设置非异步模式
	if err := llm.SetConfig(`{"async":false}`); err != nil {
		fmt.Printf("Error setting async: %v\n", err)
	}

	// 执行评估
	promptFile := args[1]
	eval(llm, promptFile, maxTokenNumber)
}
