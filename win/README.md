在Windows编译过程中遇到了MNN库的链接错误，原因是Windows(Visual-Studio)的cl.exe编译的lib与gcc编译的.a不兼容。
这里通过cl编译一个标准的DLL来做桥接：

### 1. 下载 MNN-Go 库到你的项目目录

首先需要进入Visual-Studio的dos环境：在开始菜单找到类似"x64 Native Tools Command Prompt for VS 20xx"的菜单项点击进入cmd环境。
然后下载 MNN-Go 库到你的项目目录下：

```bash
# 进入你的项目目录,克隆 MNN-Go 仓库到你的项目内
git clone https://github.com/axcom/mnn.git
cd mnn\win
```

### 2. 解压 reimp.zip

解压出reimp.exe文件到当前目录，或者拷贝到MinGW的bin目录下。

### 3. 编译 libmnn.dll 

设置dos环境变量**`MNN_ROOT`**(或编辑build.bat文件),将其设置为你的MNN的安装目录。如：
```
	set MNN_ROOT=D:\MNN
```
然后运行 build.bat 批处理程序(注意当前目录是在win目录下)。
执行该批处理指令后，当前项目mnn目录内容变更为如下：
```
mnn
├── MNN_C.h                 # MNN_C 的 C API 头文件
├── MNN_Express.h      # MNN_Express 的 C API 头文件
├── llm_c.h                      # LLM C API 头文件
├── llm.go                       # LLM 功能的 Go 绑定（llm_demo参考示例）
├── mnn_c.go                 # MNN_C 的 Go 绑定
├── mnn_express.go     # MNN_Express 的 Go 绑定
└── win		            # MNN 的 C++ 实现
	├── MNN_C.cpp               # MNN_C 的 C++ 实现
	├── MNN_Express.cpp   # MNN_Express 的 C++ 实现
	└── llm_c.cpp                   # LLM C++ 实现
├── libmnn.dll         # 编译生成的桥接dll文件
├── libmnn.a           # MinGW工具生成的供gcc链接的.a文件
└── MNN.lib            # 从MNN的build目录复制过来的lib文件
```

### 3. 修改项目的go.mod文件

修改项目的go.mod文件，添加以下内容(将MNN-Go库指向你下载的mnn目录)：
```
require (
	github.com/axcom/mnn v0.0.0-00010101000000-000000000000
)
replace github.com/axcom/mnn => ./mnn
```

### 4. 编译你的项目
```
rem 回到你的项目目录
cd ..

rem 编译go代码并链接.a(#cgo LDFLAGS: libmnn.a -lstdc++)
go build -ldflags="-s -w"
```

### 5. 复制 dll 到项目exe输出目录 
```
copy /y ./mnn/libmnn.dll .
copy /y %MNN_ROOT%\build\release\MNN.dll .
```
复制这2个dll到你的项目exe目录后，你的exe程序就可以正常运行了。


## 注意事项

在go build编译时，llm_c.go、mnn_c.go等的#cgo 设置中，目录.指的是当前执行编译时的目录。
如果提示找不到libmnn.a的话，试着调整libmnn.a的引入目录。(如：mnn/libmnn.a)
