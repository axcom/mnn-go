//
//  Interpreter_c.h
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_Interpreter_c_h
#define MNN_Interpreter_c_h

#include <stddef.h>
#include <stdint.h>
#include "MNN/HalideRuntime.h"
#include "MNN/MNNForwardType.h"
#include "ErrorCode_c.h"
#include "Tensor_c.h"

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

struct MNN_ScheduleConfig_Path;
struct MNN_ScheduleConfig;
typedef struct MNN_RuntimeInfo MNN_RuntimeInfo;
typedef struct MNN_OperatorInfo MNN_OperatorInfo;
typedef struct MNN_Session MNN_Session;
typedef struct MNN_Interpreter MNN_Interpreter;
struct MNN_Backend;
struct MNN_Runtime;

// Backend config
typedef enum MNN_MemoryMode {
    MNN_MEMORY_NORMAL = 0,
    MNN_MEMORY_HIGH,
    MNN_MEMORY_LOW
} MNN_MemoryMode;

typedef enum MNN_Gpu_PowerMode{
    MNN_POWER_NORMAL = 0,
    MNN_POWER_HIGH,
    MNN_POWER_LOW
} MNN_Gpu_PowerMode;

typedef enum MNN_Gpu_PrecisionMode{
    MNN_PRECISION_NORMAL = 0,
    MNN_PRECISION_HIGH,
    MNN_PRECISION_LOW,
    MNN_PRECISION_LOW_BF16
} MNN_Gpu_PrecisionMode;

typedef struct MNN_BackendConfig {
    MNN_MemoryMode memory;
    MNN_Gpu_PowerMode power;
    MNN_Gpu_PrecisionMode precision;
    //union {
    //    void* sharedContext;
    //    size_t flags;
    //};
    size_t flags;
} MNN_BackendConfig;

/** std::vector<std::string> **/
typedef struct StringArray {
    const char** data;  // 字符串指针数组
    size_t size;        // 数组长度
} StringArray;

MNN_C_API void freeStringArray(StringArray sa);

typedef enum MNN_Path_Mode {
        /**
         * Op Mode
         * - inputs means the source op, can NOT be empty.
         * - outputs means the sink op, can be empty.
         * The path will start from source op, then flow when encounter the sink op.
         * The sink op will not be compute in this path.
         */
        MNN_PATH_MODE_OP = 0,

        /**
         * Tensor Mode
         * - inputs means the inputs tensors, can NOT be empty.
         * - outputs means the outputs tensors, can NOT be empty.
         * It will find the pipeline that compute outputs from inputs.
         */
        MNN_PATH_MODE_TENSOR = 1
} MNN_Path_Mode;        

/** ScheduleConfig Path */
typedef struct MNN_ScheduleConfig_Path {
    /** inputs */
    //const char** inputs;
    //int inputsCount;
    StringArray inputs;
    /** outputs */
    //const char** outputs;
    //int outputsCount;
    StringArray outputs;

    /** running mode */
    MNN_Path_Mode mode;
} MNN_ScheduleConfig_Path;

/** session schedule config */
typedef struct MNN_ScheduleConfig {
    /** which tensor should be kept */
    //const char** saveTensors;
    //int saveTensorsCount;
    StringArray saveTensors;
    /** forward type */
    MNNForwardType type_;
    /** CPU:number of threads in parallel , Or GPU: mode setting*/
    int numThread;
    /*union {
        int numThread;
        int mode;
    };*/

    /** subpath to run */
    struct MNN_ScheduleConfig_Path path;

    /** backup backend used to create execution when desinated backend do NOT support any op */
    MNNForwardType backupType;

    /** extra backend config */
    struct MNN_BackendConfig* backendConfig;
} MNN_ScheduleConfig;

/** Session Mode */
enum MNN_SessionMode {
    MNN_SESSION_MODE_DEBUG = 0,
    MNN_SESSION_MODE_RELEASE = 1,

    MNN_SESSION_MODE_INPUT_INSIDE = 2,
    MNN_SESSION_MODE_INPUT_USER = 3,

    MNN_SESSION_MODE_OUTPUT_INSIDE = 4,
    MNN_SESSION_MODE_OUTPUT_USER = 5,

    MNN_SESSION_MODE_RESIZE_DIRECT = 6,
    MNN_SESSION_MODE_RESIZE_DEFER = 7,

    MNN_SESSION_MODE_BACKEND_FIX = 8, // Use the backend user set, when not support use default backend
    MNN_SESSION_MODE_BACKEND_AUTO = 9, // Auto Determine the Op type by MNN

    MNN_SESSION_MODE_MEMORY_COLLECT = 10, // Recycle static memory when session resize in case memory explosion 
    MNN_SESSION_MODE_MEMORY_CACHE = 11, // Cache the static memory for next forward usage

    MNN_SESSION_MODE_CODEGEN_DISABLE = 12, // Disable codegen in case extra build codegen cost
    MNN_SESSION_MODE_CODEGEN_ENABLE = 13, // Enable codegen
    
    MNN_SESSION_MODE_RESIZE_CHECK = 14, // Open Trace for resize
    MNN_SESSION_MODE_RESIZE_FIX = 15, // Apply Resize Optimization
    
    MNN_SESSION_MODE_MODULE_FORWARD_SEPARATE = 16,
    MNN_SESSION_MODE_MODULE_FORWARD_COMBINE = 17,
};

/** Hint Mode */
enum MNN_HintMode {
    // Max Op number for async tuning
    MNN_HINT_MODE_MAX_TUNING_NUMBER = 0,
    // Strictly check model file or not, default 1. if set 0, will not check model file valid/invalid
    MNN_HINT_MODE_STRICT_CHECK_MODEL = 1,
    MNN_HINT_MODE_MEM_ALLOCATOR_TYPE = 2,
    // Winograd unit candidates count, default 3. if set 0, will use less unit candidates for less memory at the expense of performance.
    MNN_HINT_MODE_WINOGRAD_MEMORY_LEVEL = 3,

    // Geometry Compute option, default is 0xFFFF
    MNN_HINT_MODE_GEOMETRY_COMPUTE_MASK = 4,

    // default 0
    // 1: For general convolution, use one scale&zeropoint to quant.
    // 2: use block-quant for input data.
    MNN_HINT_MODE_DYNAMIC_QUANT_OPTIONS = 5,

    // For Mobile CPU with big-litter core, set decrease rate to let MNN divide task differential by CPU's performance
    // 0-100, 50 means litter core has 50% capacity of large core
    // Default is 50
    MNN_HINT_MODE_CPU_LITTLECORE_DECREASE_RATE = 6,

    // 0: Do not quantize
    // 1: Only quantize key, use int8 asymmetric quantization 
    // 2: Only quantize value, use fp8 quantization
    // 3: quantize both key and value
    // 4: quantize query, key and value, and use gemm int8 kernel to compute K*V
    MNN_HINT_MODE_QKV_QUANT_OPTIONS = 7,

    // size limit of kvcache in memory (for a single layer)
    // if the size of kvcache exceeds the limit, it will be moved to disk
    MNN_HINT_MODE_KVCACHE_SIZE_LIMIT = 8,
    // Op encoder number for commit
    MNN_HINT_MODE_OP_ENCODER_NUMBER_FOR_COMMIT = 9,

    // KVCache Info
    MNN_HINT_MODE_KVCACHE_INFO = 10,
    // mmap allocate file size, KB
    MNN_HINT_MODE_MMAP_FILE_SIZE = 11,
    MNN_HINT_MODE_USE_CACHED_MMAP = 12,
    
    // Multi-Thread Load module, default is 0 (don't use other Thread)
    MNN_HINT_MODE_INIT_THREAD_NUMBER = 13,

    // Used CPU ids
    MNN_HINT_MODE_CPU_CORE_IDS = 14,

    // set CPU threads to use when supports Arm sme2
    MNN_HINT_MODE_CPU_SME2_INSTRUCTIONS = 15,

    // Enable KleidiAI
    MNN_HINT_MODE_CPU_ENABLE_KLEIDIAI = 16
};

/** External Path Type */
enum MNN_ExternalPathType {
    // Path of the kvcache directory
    MNN_EXTERNAL_PATH_TYPE_KVCACHE_DIR = 0,
    
    // Mid Buffer Cache File
    MNN_EXTERNAL_PATH_TYPE_FEATUREMAP_DIR = 1,

    // Weight Buffer Cache File
    MNN_EXTERNAL_PATH_TYPE_WEIGHT_DIR = 2,

    // Path of the NPU Model directory
    MNN_EXTERNAL_PATH_TYPE_NPU_FILE_DIR = 3,

    // Other types ...
};

/** Geometry Compute Mask */
#define MNN_GEOMETRY_COMPUTE_MASK_FUSEREGION (1 << 0)
#define MNN_GEOMETRY_COMPUTE_MASK_FUSEREGION_MULTI (1 << 1)
#define MNN_GEOMETRY_COMPUTE_MASK_USELOOP (1 << 2)
#define MNN_GEOMETRY_COMPUTE_MASK_OPENCACHE (1 << 3)
#define MNN_GEOMETRY_COMPUTE_MASK_ALL 0xFFFF

/** Session Info Code */
enum MNN_SessionInfoCode {
    // memory session used in MB, float
    MNN_SESSION_INFO_CODE_MEMORY = 0,

    // float operation needed in session in M, float
    MNN_SESSION_INFO_CODE_FLOPS = 1,

    // Backends in session in M, int*, length >= 1 + number of configs when create session 
    MNN_SESSION_INFO_CODE_BACKENDS = 2,

    // Resize Info, int* , the mean different from API
    MNN_SESSION_INFO_CODE_RESIZE_STATUS = 3,
    
    // Mode / NumberThread, int
    MNN_SESSION_INFO_CODE_THREAD_NUMBER = 4,

    MNN_SESSION_INFO_CODE_ALL
};

/** 回调函数类型（替代std::function） */
// before/after回调：返回true继续执行，false终止/跳过
typedef int (*MNN_TensorCallBack)(const MNN_Tensor** tensors, size_t tensorCount, const char* opName, void* userData);
typedef int (*MNN_TensorCallBackWithInfo)(const MNN_Tensor** tensors, size_t tensorCount, const MNN_OperatorInfo* info, void* userData);
// 定义C回调的上下文结构体
struct MNN_CallbackContext {
    MNN_TensorCallBack callback;  // C函数指针
    void* userData;               // 用户数据
};
struct MNN_CallbackWithInfoContext {
    MNN_TensorCallBackWithInfo callback;  // C回调
    void* userData;                       // 用户数据
};

MNN_C_API const char* MNN_getVersion();
MNN_C_API struct MNN_Interpreter* MNN_Interpreter_createFromFile(const char* file);
MNN_C_API struct MNN_Interpreter* MNN_Interpreter_createFromBuffer(const void* buffer, size_t size);
MNN_C_API void MNN_Interpreter_destroy(struct MNN_Interpreter* net);
MNN_C_API void MNN_Interpreter_setSessionMode(struct MNN_Interpreter* net, enum MNN_SessionMode mode);
MNN_C_API void MNN_Interpreter_setCacheFile(struct MNN_Interpreter* net, const char* cacheFile, size_t keySize);
MNN_C_API void MNN_Interpreter_setExternalFile(struct MNN_Interpreter* net, const char* file, size_t flag);
MNN_C_API MNN_ErrorCode MNN_Interpreter_updateCacheFile(struct MNN_Interpreter* net, struct MNN_Session* session, int flag);
MNN_C_API void MNN_Interpreter_setSessionHint(struct MNN_Interpreter* net, enum MNN_HintMode hint, int value);
MNN_C_API void MNN_Interpreter_setSessionHintArray(struct MNN_Interpreter* net, enum MNN_HintMode hint, int* value, size_t size);
MNN_C_API struct MNN_Session* MNN_Interpreter_createSession(struct MNN_Interpreter* net, const struct MNN_ScheduleConfig* config);
MNN_C_API struct MNN_Session* MNN_Interpreter_createSessionWithRuntime(MNN_Interpreter* net, const struct MNN_ScheduleConfig* config, const MNN_RuntimeInfo* runtime);
MNN_C_API struct MNN_Session* MNN_Interpreter_createMultiPathSession(MNN_Interpreter* net, const struct MNN_ScheduleConfig* configs, int configsCount);
MNN_C_API struct MNN_Session* MNN_Interpreter_createMultiPathSessionWithRuntime(MNN_Interpreter* net, const struct MNN_ScheduleConfig* configs, int configsCount, const MNN_RuntimeInfo* runtime);
MNN_C_API MNN_BOOL MNN_Interpreter_releaseSession(struct MNN_Interpreter* net, struct MNN_Session* session);
MNN_C_API void MNN_Interpreter_resizeSession(struct MNN_Interpreter* net, struct MNN_Session* session);
MNN_C_API void MNN_Interpreter_resizeSessionEx(struct MNN_Interpreter* net, struct MNN_Session* session, int needRelloc);
MNN_C_API void MNN_Interpreter_releaseModel(struct MNN_Interpreter* net);
MNN_C_API void MNN_Interpreter_getModelBuffer(const struct MNN_Interpreter* net, const void** buffer, size_t* size);
MNN_C_API const char* MNN_Interpreter_getModelVersion(const struct MNN_Interpreter* net);
MNN_C_API MNN_ErrorCode MNN_Interpreter_updateSessionToModel(struct MNN_Interpreter* net, struct MNN_Session* session);
MNN_C_API MNN_ErrorCode MNN_Interpreter_runSession(const struct MNN_Interpreter* net, struct MNN_Session* session);
MNN_C_API struct MNN_Tensor* MNN_Interpreter_getSessionInput(struct MNN_Interpreter* net, const struct MNN_Session* session, const char* name);
MNN_C_API struct MNN_Tensor* MNN_Interpreter_getSessionOutput(struct MNN_Interpreter* net, const struct MNN_Session* session, const char* name);
MNN_C_API MNN_BOOL MNN_Interpreter_getSessionInfo(struct MNN_Interpreter* net, const struct MNN_Session* session, enum MNN_SessionInfoCode code, void* ptr);
MNN_C_API void MNN_Interpreter_resizeTensor(struct MNN_Interpreter* net, struct MNN_Tensor* tensor, const int* dims, int dimsCount);
MNN_C_API void MNN_Interpreter_resizeTensor4D(struct MNN_Interpreter* net, struct MNN_Tensor* tensor, int batch, int channel, int height, int width);
MNN_C_API const struct MNN_Backend* MNN_Interpreter_getBackend(const struct MNN_Interpreter* net, const struct MNN_Session* session, const struct MNN_Tensor* tensor);
MNN_C_API const char* MNN_Interpreter_bizCode(const struct MNN_Interpreter* net);
MNN_C_API const char* MNN_Interpreter_uuid(const struct MNN_Interpreter* net);


/*
// 1. 跨平台导出宏
#include <stddef.h>   // 定义 size_t
#include <stdbool.h>  // 定义 bool
#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif
// 2. 前向声明结构体，避免类型未定义
struct MNN_Tensor;
struct MNN_OperatorInfo;
// 3. 粘贴与CGO自动生成匹配的原型（声明Go函数的原型,与Go的函数签名一致。）
EXPORT bool BasicBeforeCallback(struct MNN_Tensor** tensors, size_t tensorCount, char* opName, void* userData);
EXPORT bool BasicAfterCallback(struct MNN_Tensor** tensors, size_t tensorCount, char* opName, void* userData);
EXPORT int BeforeCallbackWithInfo(struct MNN_Tensor** tensors, size_t tensorCount, MNN_OperatorInfo* info, void* userData);
EXPORT int AfterCallbackWithInfo(struct MNN_Tensor** tensors, size_t tensorCount, MNN_OperatorInfo* info, void* userData);
// 4. Go函数代码行前加上export（使生成函数符合C规范）
//export BasicBeforeCallback
*/
MNN_C_API MNN_ErrorCode MNN_Interpreter_runSessionWithCallBack(const MNN_Interpreter* interpreter, const MNN_Session* session,
                                                    MNN_TensorCallBack before, MNN_TensorCallBack after,
                                                    MNN_BOOL sync, void* userData);
MNN_C_API MNN_ErrorCode MNN_Interpreter_runSessionWithCallBackInfo(const MNN_Interpreter* interpreter, const MNN_Session* session,
                                                        MNN_TensorCallBackWithInfo before, MNN_TensorCallBackWithInfo after,
                                                        MNN_BOOL sync, void* userData);

// 张量名称-指针对（用于传递std::map的内容）
typedef struct {
    const char* name;
    MNN_Tensor* tensor;
} MNN_NamedTensor;
// 张量列表（包含数量和数组）
typedef struct {
    int count;
    MNN_NamedTensor* tensors;
} MNN_NamedTensorList;
// 获取所有输入张量（返回封装后的列表）
MNN_C_API MNN_NamedTensorList MNN_Interpreter_GetSessionInputAll(MNN_Interpreter* interpreter, MNN_Session* session);
// 获取所有输出张量（返回封装后的列表）
MNN_C_API MNN_NamedTensorList MNN_Interpreter_GetSessionOutputAll(MNN_Interpreter* interpreter, MNN_Session* session);
// 释放张量列表的内存
MNN_C_API void MNN_NamedTensorList_Free(MNN_NamedTensorList list);


// Runtime management
MNN_C_API MNN_RuntimeInfo* MNN_Interpreter_createRuntime(const struct MNN_ScheduleConfig* configs, int configsCount);
MNN_C_API void MNN_RuntimeInfo_destroy(MNN_RuntimeInfo* runtime);

// OperatorInfo
MNN_C_API const char* MNN_OperatorInfo_name(const MNN_OperatorInfo* info);
MNN_C_API const char* MNN_OperatorInfo_type(const MNN_OperatorInfo* info);
MNN_C_API float MNN_OperatorInfo_flops(const MNN_OperatorInfo* info);

#ifdef __cplusplus
}
#endif

#endif /* MNN_Interpreter_c_h */