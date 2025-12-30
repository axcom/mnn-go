package mnn

/*
#cgo CFLAGS: -I.
#cgo CXXFLAGS: -I./ -std=c++11
#cgo LDFLAGS: -L./
#cgo linux LDFLAGS: -lMNN -lstdc++
#cgo windows LDFLAGS: libmnn.a -lstdc++

#include <stdlib.h>
#include "Interpreter_c.h"

// Define bool type for C compatibility
#ifdef __cplusplus
typedef bool MNN_BOOL;
#else
typedef unsigned char MNN_BOOL;
#define true 1
#define false 0
#endif
*/
import "C"
import "unsafe"

// 转换工具函数
func B2C(b bool) C.MNN_BOOL {
	if b {
		return C.MNN_BOOL(1)
	}
	return C.MNN_BOOL(0)
}

func B2Go(b C.MNN_BOOL) bool {
	return b != 0
}

/* C侧 std::vector<std::string>
typedef struct StringArray {
    const char** data;  // 字符串指针数组
    size_t size;        // 数组长度
} StringArray;
// 将C的StringArray转换为C++的std::vector<std::string>
std::vector<std::string> StringArrayToVector(const StringArray& sa) {
    std::vector<std::string> vec;
    vec.reserve(sa.size); // 预分配内存
    for (size_t i = 0; i < sa.size; ++i) {
        vec.emplace_back(sa.data[i]); // 深拷贝C字符串到std::string
    }
    return vec;
}

// 将C++的std::vector<std::string>转换为C的StringArray
StringArray VectorToStringArray(const std::vector<std::string>& vec) {
    StringArray sa;
    sa.size = vec.size();
    sa.data = static_cast<const char**>(malloc(vec.size() * sizeof(char*)));
    for (size_t i = 0; i < vec.size(); ++i) {
        sa.data[i] = strdup(vec[i].c_str()); // 深拷贝std::string到C字符串
    }
    return sa;
}
*/

// ToCStringArray []string转换为C侧_StringArray
func ToCStringArray(sa []string) C.StringArray {
	if sa == nil || len(sa) == 0 {
		return C.StringArray{data: nil, size: 0}
	}

	// 1. 计算指针数组的字节数：每个元素是*const char，占一个指针大小
	ptrSize := C.size_t(unsafe.Sizeof((*C.char)(nil)))
	bufSize := C.size_t(len(sa)) * ptrSize

	// 2. 分配C堆内存存储char**数组，返回的是void*，需转为**C.char
	cData := C.malloc(bufSize)
	if cData == nil {
		panic("malloc failed for StringArray data")
	}

	// 3. 将void*转换为指向C.char指针数组的指针，再映射为Go切片方便操作
	// 用超大数组的指针来映射，限制切片长度为sa的长度
	cStrArray := (*[1 << 30]*C.char)(cData)
	cStrSlice := cStrArray[:len(sa):len(sa)]

	// 4. 遍历Go字符串，转换为C字符串并赋值到指针数组
	for i, s := range sa {
		cStr := C.CString(s) // C.CString分配的内存需手动释放
		cStrSlice[i] = cStr
	}

	// 5. 将char**转换为const char**，封装为StringArray
	return C.StringArray{
		data: (**C.char)(unsafe.Pointer(cData)),
		size: C.size_t(len(sa)),
	}
}

// FreeCStringArray 释放转换后的StringArray内存
func FreeCStringArray(sa *C.StringArray) {
	if sa == nil {
		return
	}
	C.freeStringArray(*sa)
}
