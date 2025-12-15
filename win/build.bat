rem 设置 MNN 根目录
if not defined MNN_ROOT set MNN_ROOT=f:\openAI\mnn

rem 设置 CGO 环境变量
set CGO_CFLAGS="-I%MNN_ROOT%\include"
set CGO_CXXFLAGS="-I%MNN_ROOT%\include -I%MNN_ROOT%\transformers\llm\engine\include"
set CGO_LDFLAGS="-L%MNN_ROOT%\build -L%MNN_ROOT%\build\express"

rem 移cpp文件到win目录
move /y ..\*.cpp .

rem 进入"x64 Native Tools Command Prompt for VS 20xx"环境
rem 先用cl编译出.obj
cl /c /MD /DWIN64 /DMNN_C_EXPORTS /I"%MNN_ROOT%\include" /I".." mnn_c.cpp
cl /c /MD /DWIN64 /DMNN_C_EXPORTS /I"%MNN_ROOT%\include" /I".." mnn_express.cpp
cl /c /MD /DWIN64 /DMNN_C_EXPORTS /I"%MNN_ROOT%\include" /I".." llm_c.cpp /I"%MNN_ROOT%\transformers\llm\engine\include" /EHsc

rem 链接到libmnn.dll
cl /LD /MD /DWIN64 /I".." mnn_c.obj mnn_express.obj llm_c.obj -link %MNN_ROOT%\build\release\MNN.lib /implib:..\libmnn.lib -out:..\libmnn.dll

rem 回退到mnn目录
cd ..

rem 通过MinGW的reimp工具生成libmnn.def后，再生成libmnn.a
reimp -d libmnn.lib
dlltool -d libmnn.def -l libmnn.a -D libmnn.dll

rem 复制 MNN.lib 到mnn目录 
copy /y %MNN_ROOT%\build\release\MNN.lib .

rem 复制 dll 到你的项目编译输出目录(修改..为你的项目exe的输出目录)
copy /y libmnn.dll ..
copy /y %MNN_ROOT%\build\release\MNN.dll ..