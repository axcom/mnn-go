# MNN Go 绑定

这是 MNN (Mobile Neural Network) 框架的 Go 语言绑定，提供了对 MNN API 的 Go 封装。

## 目录结构

```
.
├── mnn                # MNN.dll 的 Go 绑定
├── expr               # MNN_Express.dll 的 Go 绑定
├── llm                # LLM.dll 的 Go 绑定
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
cmake -DMNN_BUILD_TEST=OFF -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_CONVERTER=OFF ..
make -j8
```

### 2. 配置环境变量

确保 MNN 库的头文件和库文件可以被找到：

```bash
# 设置 MNN 根目录
export MNN_ROOT=$(pwd)/..

# 设置 CGO 环境变量
export CGO_CFLAGS="-I$MNN_ROOT/include"
export CGO_CXXFLAGS="-I$MNN_ROOT/include"
export CGO_LDFLAGS="-L$MNN_ROOT/build -lMNN -lMNN_Express -lstdc++"
```

## 注意事项

1. 所有的 C 资源都需要手动释放，使用 `Close()` 方法或 defer 语句确保资源被正确释放。
2. 确保在编译前正确配置 MNN 库的路径。

## 许可证

MIT
