//
//  Tensor_c.h
//  MNN
//
//  Created by MNN on 2018/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_Tensor_c_h
#define MNN_Tensor_c_h

#include <stdint.h>
#include <MNN/HalideRuntime.h>
#include <MNN/MNNDefine.h>
#include <ErrorCode_c.h>

// 导出宏：Windows下导出，其他平台兼容
#ifdef MNN_C_EXPORTS
    // 仅当编译 libmnn.dll 时定义 MNN_C_EXPORTS，此时用 dllexport
    #define MNN_C_API __declspec(dllexport)
#else
    #define MNN_C_API
#endif

// Define bool type for C compatibility
#ifdef __cplusplus
typedef bool MNN_BOOL;
#else
typedef unsigned char MNN_BOOL;
#define true 1
#define false 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** dimension type used to create tensor */
enum MNN_DimensionType {
    /** for tensorflow net type. uses NHWC as data format. */
    MNN_TENSORFLOW,
    /** for caffe net type. uses NCHW as data format. */
    MNN_CAFFE,
    /** for caffe net type. uses NC4HW4 as data format. */
    MNN_CAFFE_C4
};

/** handle type */
enum MNN_HandleDataType {
    /** default handle type */
    MNN_HANDLE_NONE = 0,
    /** string handle type */
    MNN_HANDLE_STRING = 1
};

/** Tensor map type : Read or Write*/
enum MNN_MapType {
    /** map Tensor for writing data*/
    MNN_MAP_TENSOR_WRITE = 0,
    /** map Tensor for reading data*/
    MNN_MAP_TENSOR_READ = 1
};

// Forward declaration
typedef struct MNN_Tensor MNN_Tensor;

// Tensor creation and destruction
MNN_C_API struct MNN_Tensor* MNN_Tensor_Create(int dimSize, enum MNN_DimensionType type);
MNN_C_API struct MNN_Tensor* MNN_Tensor_CreateFromExisting(const struct MNN_Tensor* tensor, enum MNN_DimensionType type, MNN_BOOL allocMemory);
MNN_C_API struct MNN_Tensor* MNN_Tensor_CreateDevice(const int* shape, int shapeSize, struct halide_type_t type, enum MNN_DimensionType dimType);
MNN_C_API struct MNN_Tensor* MNN_Tensor_CreateHost(const int* shape, int shapeSize, struct halide_type_t type, void* data, enum MNN_DimensionType dimType);
MNN_C_API struct MNN_Tensor* MNN_Tensor_Clone(const struct MNN_Tensor* src, MNN_BOOL deepCopy);
MNN_C_API void MNN_Tensor_Destroy(struct MNN_Tensor* tensor);

// Tensor data operations
MNN_C_API MNN_BOOL MNN_Tensor_CopyFromHostTensor(struct MNN_Tensor* tensor, const struct MNN_Tensor* hostTensor);
MNN_C_API MNN_BOOL MNN_Tensor_CopyToHostTensor(const struct MNN_Tensor* tensor, struct MNN_Tensor* hostTensor);
MNN_C_API struct MNN_Tensor* MNN_Tensor_CreateHostTensorFromDevice(const struct MNN_Tensor* deviceTensor, MNN_BOOL copyData);

// Tensor properties access
MNN_C_API const halide_buffer_t* MNN_Tensor_Buffer(const struct MNN_Tensor* tensor);
MNN_C_API halide_buffer_t* MNN_Tensor_MutableBuffer(struct MNN_Tensor* tensor);
MNN_C_API enum MNN_DimensionType MNN_Tensor_GetDimensionType(const struct MNN_Tensor* tensor);
MNN_C_API enum MNN_HandleDataType MNN_Tensor_GetHandleDataType(const struct MNN_Tensor* tensor);
MNN_C_API void MNN_Tensor_SetType(struct MNN_Tensor* tensor, int type);
//MNN_C_API struct halide_type_t MNN_Tensor_GetType(const struct MNN_Tensor* tensor);
  MNN_C_API void MNN_Tensor_GetHalideType(const MNN_Tensor* tensor, struct halide_type_t* outType);
MNN_C_API void* MNN_Tensor_Host(const struct MNN_Tensor* tensor);
MNN_C_API uint64_t MNN_Tensor_DeviceId(const struct MNN_Tensor* tensor);

// Tensor shape and size
MNN_C_API int MNN_Tensor_Dimensions(const struct MNN_Tensor* tensor);
MNN_C_API int* MNN_Tensor_Shape(const struct MNN_Tensor* tensor, int* shapeSize);
MNN_C_API void MNN_Tensor_FreeShape(int* shape);
MNN_C_API int MNN_Tensor_Size(const struct MNN_Tensor* tensor);
MNN_C_API size_t MNN_Tensor_USize(const struct MNN_Tensor* tensor);
MNN_C_API int MNN_Tensor_ElementSize(const struct MNN_Tensor* tensor);

// Tensor dimension accessors
MNN_C_API int MNN_Tensor_Width(const struct MNN_Tensor* tensor);
MNN_C_API int MNN_Tensor_Height(const struct MNN_Tensor* tensor);
MNN_C_API int MNN_Tensor_Channel(const struct MNN_Tensor* tensor);
MNN_C_API int MNN_Tensor_Batch(const struct MNN_Tensor* tensor);
MNN_C_API int MNN_Tensor_Stride(const struct MNN_Tensor* tensor, int index);
MNN_C_API int MNN_Tensor_Length(const struct MNN_Tensor* tensor, int index);
MNN_C_API void MNN_Tensor_SetStride(struct MNN_Tensor* tensor, int index, int stride);
MNN_C_API void MNN_Tensor_SetLength(struct MNN_Tensor* tensor, int index, int length);

// Device operations
MNN_C_API MNN_BOOL MNN_Tensor_GetDeviceInfo(const struct MNN_Tensor* tensor, void* dst, int forwardType);

// Debug operations
MNN_C_API void MNN_Tensor_Print(const struct MNN_Tensor* tensor);
MNN_C_API void MNN_Tensor_PrintShape(const struct MNN_Tensor* tensor);

// GPU operations
MNN_C_API void* MNN_Tensor_Map(struct MNN_Tensor* tensor, enum MNN_MapType mtype, enum MNN_DimensionType dtype);
MNN_C_API void MNN_Tensor_Unmap(struct MNN_Tensor* tensor, enum MNN_MapType mtype, enum MNN_DimensionType dtype, void* mapPtr);
MNN_C_API int MNN_Tensor_Wait(struct MNN_Tensor* tensor, enum MNN_MapType mtype, MNN_BOOL finish);
MNN_C_API MNN_BOOL MNN_Tensor_SetDevicePtr(struct MNN_Tensor* tensor, const void* devicePtr, int memoryType);

#ifdef __cplusplus
}
#endif

#endif /* MNN_Tensor_c_h */
