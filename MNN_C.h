#ifndef MNN_C_H
#define MNN_C_H

#include <stddef.h>
#include <stdint.h>

// Define bool type for C compatibility
#ifdef __cplusplus
typedef bool MNN_BOOL;
#else
typedef unsigned char MNN_BOOL;
#define true 1
#define false 0
#endif

// 导出宏：Windows下导出，其他平台兼容
#ifdef MNN_C_EXPORTS
    // 仅当编译 libmnn.dll 时定义 MNN_C_EXPORTS，此时用 dllexport
    #define MNN_C_API __declspec(dllexport)
#else
    #define MNN_C_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct MNN_Interpreter MNN_Interpreter;
typedef struct MNN_Session MNN_Session;
typedef struct MNN_Tensor MNN_Tensor;
typedef struct MNN_ScheduleConfig MNN_ScheduleConfig;

// Forward type for C API
typedef enum {
    MNN_C_FORWARD_CPU = 0,
    MNN_C_FORWARD_METAL = 1,
    MNN_C_FORWARD_CUDA = 2,
    MNN_C_FORWARD_OPENCL = 3,
    MNN_C_FORWARD_AUTO = 4,
    MNN_C_FORWARD_NN = 5,
    MNN_C_FORWARD_OPENGL = 6,
    MNN_C_FORWARD_VULKAN = 7,
    MNN_C_FORWARD_USER_0 = 8,
    MNN_C_FORWARD_USER_1 = 9,
    MNN_C_FORWARD_USER_2 = 10,
    MNN_C_FORWARD_USER_3 = 11,
    MNN_C_FORWARD_ALL = 12,
    MNN_C_FORWARD_CPU_EXTENSION = 13
} MNN_C_ForwardType;

// ErrorCode.h
typedef enum {
    NO_ERROR = 0,
    OUT_OF_MEMORY = 1,
    NOT_SUPPORT = 2,
    NO_FILE = 3,
    INVALID_DATA = 4,
    UNKNOWN_ERROR = 5,
} MNN_ErrorCode;

// Tensor dimension type
typedef enum {
    MNN_TENSORFLOW = 0,
    MNN_CAFFE = 1,
    MNN_CAFFE_C4 = 2,
} MNN_DimensionType;

// Dimension format for Express
typedef enum {
    MNN_EXPRESS_NHWC = 0,
    MNN_EXPRESS_NC4HW4 = 1,
    MNN_EXPRESS_NCHW = 2,
} MNN_Express_DimensionFormat;

// ScheduleConfig
typedef struct MNN_ScheduleConfig {
    MNN_C_ForwardType datatype;
    int numThread;
    MNN_C_ForwardType backupType;
} MNN_ScheduleConfig;

// ImageProcess module
typedef struct MNN_CV_ImageProcess MNN_CV_ImageProcess;

// ImageFormat
typedef enum {
    MNN_CV_RGBA     = 0,
    MNN_CV_RGB      = 1,
    MNN_CV_BGR      = 2,
    MNN_CV_GRAY     = 3,
    MNN_CV_BGRA     = 4,
    MNN_CV_YCrCb    = 5,
    MNN_CV_YUV      = 6,
    MNN_CV_HSV      = 7,
    MNN_CV_XYZ      = 8,
    MNN_CV_BGR555   = 9,
    MNN_CV_BGR565   = 10,
    MNN_CV_YUV_NV21 = 11,
    MNN_CV_YUV_NV12 = 12,
    MNN_CV_YUV_I420 = 13,
    MNN_CV_HSV_FULL = 14,
} MNN_CV_ImageFormat;

// Filter
typedef enum {
    MNN_CV_NEAREST = 0,
    MNN_CV_BILINEAR = 1,
    MNN_CV_BICUBIC = 2
} MNN_CV_Filter;

// Wrap
typedef enum {
    MNN_CV_CLAMP_TO_EDGE = 0,
    MNN_CV_ZERO = 1,
    MNN_CV_REPEAT = 2
} MNN_CV_Wrap;

// ImageProcess Config
typedef struct MNN_CV_ImageProcess_Config {
    MNN_CV_Filter filterType;
    MNN_CV_ImageFormat sourceFormat;
    MNN_CV_ImageFormat destFormat;
    float mean[4];
    float normal[4];
    MNN_CV_Wrap wrap;
} MNN_CV_ImageProcess_Config;

// Interpreter functions
MNN_C_API MNN_Interpreter* MNN_Interpreter_createFromFile(const char* file);
MNN_C_API MNN_Interpreter* MNN_Interpreter_createFromBuffer(const void* buffer, size_t size);
MNN_C_API void MNN_Interpreter_destroy(MNN_Interpreter* net);
MNN_C_API MNN_Session* MNN_Interpreter_createSession(MNN_Interpreter* net, const MNN_ScheduleConfig* config);
MNN_C_API MNN_BOOL MNN_Interpreter_releaseSession(MNN_Interpreter* net, MNN_Session* session);
MNN_C_API void MNN_Interpreter_resizeSession(MNN_Interpreter* net, MNN_Session* session);
MNN_C_API void MNN_Interpreter_resizeTensor(MNN_Interpreter* net, MNN_Tensor* tensor, const int* dims, int dimSize);
MNN_C_API MNN_ErrorCode MNN_Interpreter_runSession(MNN_Interpreter* net, MNN_Session* session);
MNN_C_API MNN_Tensor* MNN_Interpreter_getSessionInput(MNN_Interpreter* net, MNN_Session* session, const char* name);
MNN_C_API MNN_Tensor* MNN_Interpreter_getSessionOutput(MNN_Interpreter* net, MNN_Session* session, const char* name);

// Tensor functions
MNN_C_API MNN_Tensor* MNN_Tensor_createHostTensorFromDevice(const MNN_Tensor* deviceTensor, MNN_BOOL copyData);
MNN_C_API MNN_Tensor* MNN_Tensor_create(const MNN_Tensor* tensor, MNN_DimensionType type, MNN_BOOL allocMemory);
MNN_C_API void MNN_Tensor_destroy(MNN_Tensor* tensor);
MNN_C_API MNN_BOOL MNN_Tensor_copyFromHostTensor(MNN_Tensor* deviceTensor, const MNN_Tensor* hostTensor);
MNN_C_API MNN_BOOL MNN_Tensor_copyToHostTensor(const MNN_Tensor* deviceTensor, MNN_Tensor* hostTensor);
MNN_C_API float* MNN_Tensor_getFloatData(const MNN_Tensor* tensor);
MNN_C_API int MNN_Tensor_elementSize(const MNN_Tensor* tensor);
MNN_C_API void MNN_Tensor_getShape(const MNN_Tensor* tensor, int* shape, int shapeSize);

// Utility functions
MNN_C_API const char* MNN_getVersion();

// ImageProcess functions
MNN_C_API MNN_CV_ImageProcess* MNN_CV_ImageProcess_create(const MNN_CV_ImageProcess_Config* config, const MNN_Tensor* dstTensor);
MNN_C_API void MNN_CV_ImageProcess_destroy(MNN_CV_ImageProcess* imageProcess);
MNN_C_API void MNN_CV_ImageProcess_setMatrix(MNN_CV_ImageProcess* imageProcess, const float* matrix); // 3x3 matrix
MNN_C_API MNN_ErrorCode MNN_CV_ImageProcess_convert(MNN_CV_ImageProcess* imageProcess, const uint8_t* source, int iw, int ih, int stride, MNN_Tensor* dest);
MNN_C_API void MNN_CV_ImageProcess_setPadding(MNN_CV_ImageProcess* imageProcess, uint8_t value);

#ifdef __cplusplus
}
#endif

#endif // MNN_C_H
