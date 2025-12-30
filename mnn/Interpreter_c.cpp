//
//  Interpreter_c.cpp
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Interpreter_c.h"
#include "MNN/Interpreter.hpp"
#include "MNN/MNNForwardType.h"
#include <vector>
#include <cstring>

using namespace MNN;

// 释放StringArray的内存（辅助函数）
void freeStringArray(StringArray sa) {
    if (sa.data != NULL) {
        // 先释放每个字符串的内存
        for (size_t i = 0; i < sa.size; i++) {
            free((void*)sa.data[i]);
        }
        // 再释放指针数组的内存
        free((void*)sa.data);
    }
}

// Interpreter functions
static std::vector<std::string> StringArrayToVector(const StringArray& strArray) {
    std::vector<std::string> vec;
    if (strArray.data == nullptr || strArray.size == 0) {
        return vec;
    }
    for (size_t i = 0; i < strArray.size; ++i) {
        if (strArray.data[i] != nullptr) { 
            vec.emplace_back(strArray.data[i]);
        }
    }
    return vec;
}

// Helper function to convert C ScheduleConfig to C++ ScheduleConfig
static ScheduleConfig convertToCppScheduleConfig(const MNN_ScheduleConfig* cConfig) {
    ScheduleConfig cppConfig;
    if (cConfig == nullptr) {
        return cppConfig; 
    }

    cppConfig.saveTensors = StringArrayToVector(cConfig->saveTensors);

    cppConfig.type = static_cast<MNNForwardType>(cConfig->type_);

    cppConfig.numThread = cConfig->numThread;

    cppConfig.path.inputs = StringArrayToVector(cConfig->path.inputs);
    cppConfig.path.outputs = StringArrayToVector(cConfig->path.outputs);
    cppConfig.path.mode = static_cast<MNN::ScheduleConfig::Path::Mode>(cConfig->path.mode);

    cppConfig.backupType = static_cast<MNNForwardType>(cConfig->backupType);

    if (cConfig->backendConfig == nullptr) {
        return cppConfig;
    }
    
    /*auto* cppBackendConfig = new MNN::BackendConfig();
    cppBackendConfig->memory = static_cast<MNN::BackendConfig::MemoryMode>(cConfig->backendConfig->memory);
	cppBackendConfig->power =  static_cast<MNN::BackendConfig::PowerMode>(cConfig->backendConfig->power);
	cppBackendConfig->precision = static_cast<MNN::BackendConfig::PrecisionMode>(cConfig->backendConfig->precision);
    cppConfig.backendConfig = cppBackendConfig;*/
    cppConfig.backendConfig = reinterpret_cast<BackendConfig*>(cConfig->backendConfig);
    return cppConfig;
}


/**
 * @brief get mnn version info.
 * @return mnn version string.
 */
const char* MNN_getVersion() {
    return getVersion();
}

/**
 * @brief create net from file.
 * @param file  given file.
 * @return created net if success, NULL otherwise.
 */
struct MNN_Interpreter* MNN_Interpreter_createFromFile(const char* file) {
    return reinterpret_cast<struct MNN_Interpreter*>(Interpreter::createFromFile(file));
}

/**
 * @brief create net from buffer.
 * @param buffer    given data buffer.
 * @param size      size of data buffer.
 * @return created net if success, NULL otherwise.
 */
struct MNN_Interpreter* MNN_Interpreter_createFromBuffer(const void* buffer, size_t size) {
    return reinterpret_cast<struct MNN_Interpreter*>(Interpreter::createFromBuffer(buffer, size));
}

/**
 * @brief destroy Interpreter
 * @param net    given Interpreter to release.
 */
void MNN_Interpreter_destroy(struct MNN_Interpreter* net) {
    Interpreter::destroy(reinterpret_cast<Interpreter*>(net));
}

/**
 * @brief The API shoud be called before create session.
 * @param net      given Interpreter.
 * @param mode      session mode
 */
void MNN_Interpreter_setSessionMode(struct MNN_Interpreter* net, enum MNN_SessionMode mode) {
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    cppNet->setSessionMode(static_cast<Interpreter::SessionMode>(mode));
}

/**
 * @brief The API shoud be called before create session.
 * If the cache exist, try to load cache from file.
 * After createSession, try to save cache to file.
 * @param net      given Interpreter.
 * @param cacheFile      cache file name
 * @param keySize        depercerate, for future use.
 */
void MNN_Interpreter_setCacheFile(struct MNN_Interpreter* net, const char* cacheFile, size_t keySize) {
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    cppNet->setCacheFile(cacheFile, keySize);
}

/**
 * @brief The API shoud be called before create session.
 * @param net      given Interpreter.
 * @param file      external data file name
 * @param flag        depercerate, for future use.
 */
void MNN_Interpreter_setExternalFile(struct MNN_Interpreter* net, const char* file, size_t flag) {
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    cppNet->setExternalFile(file, flag);
}

/**
 * @brief The API shoud be called after last resize session.
 * If resize session generate new cache info, try to rewrite cache file.
 * If resize session do not generate any new cache info, just do nothing.
 * @param net      given Interpreter.
 * @param session    given session
 * @param flag   Protected param, not used now 
 */
MNN_ErrorCode MNN_Interpreter_updateCacheFile(struct MNN_Interpreter* net, struct MNN_Session* session, int flag) {
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    auto cppSession = reinterpret_cast<Session*>(session);
    return static_cast<MNN_ErrorCode>(cppNet->updateCacheFile(cppSession, flag));
}

/**
 * @brief The API shoud be called before create session.
 * @param net      given Interpreter.
 * @param hint      Hint type
 * @param value     Hint value
 */
void MNN_Interpreter_setSessionHint(struct MNN_Interpreter* net, enum MNN_HintMode hint, int value) {
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    cppNet->setSessionHint(static_cast<Interpreter::HintMode>(hint), value);
}

/**
 * @brief The API shoud be called before create session.
 * @param net      given Interpreter.
 * @param hint      Hint type
 * @param value     Hint value
 * @param size      Hint value size(when use a ptr)
 */
void MNN_Interpreter_setSessionHintArray(struct MNN_Interpreter* net, enum MNN_HintMode hint, int* value, size_t size) {
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    cppNet->setSessionHint(static_cast<Interpreter::HintMode>(hint), value, size);
}

/**
 * @brief create session with schedule config. created session will be managed in net.
 * @param net      given Interpreter.
 * @param config session schedule config.
 * @return created session if success, NULL otherwise.
 */
struct MNN_Session* MNN_Interpreter_createSession(struct MNN_Interpreter* net, const struct MNN_ScheduleConfig* config) {
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    auto cppConfig = convertToCppScheduleConfig(config);
    return reinterpret_cast<struct MNN_Session*>(cppNet->createSession(cppConfig));
}

MNN_Session* MNN_Interpreter_createSessionWithRuntime(MNN_Interpreter* net, const MNN_ScheduleConfig* config, const MNN_RuntimeInfo* runtime) {
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    auto cppConfig = convertToCppScheduleConfig(config);
    auto cppRuntime = *reinterpret_cast<const RuntimeInfo*>(runtime);
    return reinterpret_cast<struct MNN_Session*>(cppNet->createSession(cppConfig, cppRuntime));
}

MNN_Session* MNN_Interpreter_createMultiPathSession(MNN_Interpreter* net, const MNN_ScheduleConfig* configs, int configsCount) {
    std::vector<ScheduleConfig> cppConfigs;
    cppConfigs.reserve(configsCount);
    
    for (int i = 0; i < configsCount; i++) {
        const MNN_ScheduleConfig* config = &configs[i];
        ScheduleConfig cppConfig = convertToCppScheduleConfig(config);
        cppConfigs.push_back(cppConfig);
    }
    
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    return reinterpret_cast<struct MNN_Session*>(cppNet->createMultiPathSession(cppConfigs));
}

MNN_Session* MNN_Interpreter_createMultiPathSessionWithRuntime(MNN_Interpreter* net, const MNN_ScheduleConfig* configs, int configsCount, const MNN_RuntimeInfo* runtime) {
    std::vector<ScheduleConfig> cppConfigs;
    cppConfigs.reserve(configsCount);
    
    for (int i = 0; i < configsCount; i++) {
        const MNN_ScheduleConfig* config = &configs[i];
        ScheduleConfig cppConfig = convertToCppScheduleConfig(config);
        cppConfigs.push_back(cppConfig);
    }
    
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    auto cppRuntime = *reinterpret_cast<const RuntimeInfo*>(runtime);
    return reinterpret_cast<struct MNN_Session*>(cppNet->createMultiPathSession(cppConfigs, cppRuntime));
}

/**
 * @brief release session.
 * @param net      given Interpreter.
 * @param session   given session.
 * @return true if given session is held by net and is freed.
 */
MNN_BOOL MNN_Interpreter_releaseSession(struct MNN_Interpreter* net, struct MNN_Session* session) {
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    auto cppSession = reinterpret_cast<Session*>(session);
    return cppNet->releaseSession(cppSession);
}

/**
 * @brief call this function to get tensors ready. output tensor buffer (host or deviceId) should be retrieved
 *        after resize of any input tensor.
 * @param net      given Interpreter.
 * @param session given session.
 */
void MNN_Interpreter_resizeSession(struct MNN_Interpreter* net, struct MNN_Session* session) {
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    auto cppSession = reinterpret_cast<Session*>(session);
    cppNet->resizeSession(cppSession);
}

/**
 * @brief call this function to get tensors ready. output tensor buffer (host or deviceId) should be retrieved
 *        after resize of any input tensor.
 * @param net      given Interpreter.
 * @param session given session.
 * @param needRelloc, 1 means need realloc.
 */
void MNN_Interpreter_resizeSessionEx(struct MNN_Interpreter* net, struct MNN_Session* session, int needRelloc) {
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    auto cppSession = reinterpret_cast<Session*>(session);
    cppNet->resizeSession(cppSession, needRelloc);
}

/**
 * @brief call this function if don't need resize or create session any more, it will save a few memory that equal
 * to the size of model buffer
 * @param net      given Interpreter.
 */
void MNN_Interpreter_releaseModel(struct MNN_Interpreter* net) {
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    cppNet->releaseModel();
}

/**
 * @brief Get the model buffer for user to save
 * @param net      given Interpreter.
 * @param buffer    output buffer pointer.
 * @param size      output buffer size.
 */
void MNN_Interpreter_getModelBuffer(const struct MNN_Interpreter* net, const void** buffer, size_t* size) {
    auto cppNet = reinterpret_cast<const Interpreter*>(net);
    auto cppBuffer = cppNet->getModelBuffer();
    if (buffer) *buffer = cppBuffer.first;
    if (size) *size = cppBuffer.second;
}

/**
 * @brief Get the model's version info.
 * @param net      given Interpreter.
 * @return const char* of model's version info like "2.0.0";
 * If model is not loaded or model no version info, return "version info not found".
 */
const char* MNN_Interpreter_getModelVersion(const struct MNN_Interpreter* net) {
    auto cppNet = reinterpret_cast<const Interpreter*>(net);
    return cppNet->getModelVersion();
}

/**
 * @brief update Session's Tensor to model's Const Op
 * @param net      given Interpreter.
 * @param session   given session.
 * @return result of running.
 */
MNN_ErrorCode MNN_Interpreter_updateSessionToModel(struct MNN_Interpreter* net, struct MNN_Session* session) {
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    auto cppSession = reinterpret_cast<Session*>(session);
    return static_cast<MNN_ErrorCode>(cppNet->updateSessionToModel(cppSession));
}

/**
 * @brief run session.
 * @param net      given Interpreter.
 * @param session   given session.
 * @return result of running.
 */
MNN_ErrorCode MNN_Interpreter_runSession(const struct MNN_Interpreter* net, struct MNN_Session* session) {
    auto cppNet = reinterpret_cast<const Interpreter*>(net);
    auto cppSession = reinterpret_cast<Session*>(session);
    return static_cast<MNN_ErrorCode>(cppNet->runSession(cppSession));
}

/**
 * @brief get input tensor for given name.
 * @param net      given Interpreter.
 * @param session   given session.
 * @param name      given name. if NULL, return first input.
 * @return tensor if found, NULL otherwise.
 */
struct MNN_Tensor* MNN_Interpreter_getSessionInput(struct MNN_Interpreter* net, const struct MNN_Session* session, const char* name) {
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    auto cppSession = reinterpret_cast<const Session*>(session);
    return reinterpret_cast<struct MNN_Tensor*>(cppNet->getSessionInput(cppSession, name));
}

/**
 * @brief get output tensor for given name.
 * @param net      given Interpreter.
 * @param session   given session.
 * @param name      given name. if NULL, return first output.
 * @return tensor if found, NULL otherwise.
 */
struct MNN_Tensor* MNN_Interpreter_getSessionOutput(struct MNN_Interpreter* net, const struct MNN_Session* session, const char* name) {
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    auto cppSession = reinterpret_cast<const Session*>(session);
    return reinterpret_cast<struct MNN_Tensor*>(cppNet->getSessionOutput(cppSession, name));
}

/**
 * @brief get session info
 * @param net      given Interpreter.
 * @param session   given session.
 * @param code      given info code.
 * @param ptr     given info ptr, see SessionInfoCode for detail
 * @return true if support the code, false otherwise.
 */
MNN_BOOL MNN_Interpreter_getSessionInfo(struct MNN_Interpreter* net, const struct MNN_Session* session, enum MNN_SessionInfoCode code, void* ptr) {
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    auto cppSession = reinterpret_cast<const Session*>(session);
    return cppNet->getSessionInfo(cppSession, static_cast<Interpreter::SessionInfoCode>(code), ptr);
}

/**
 * @brief resize given tensor.
 * @param net      given Interpreter.
 * @param tensor    given tensor.
 * @param dims      new dims.
 * @param dimsCount number of dims.
 */
void MNN_Interpreter_resizeTensor(struct MNN_Interpreter* net, struct MNN_Tensor* tensor, const int* dims, int dimsCount) {
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    auto cppTensor = reinterpret_cast<Tensor*>(tensor);
    std::vector<int> cppDims(dims, dims + dimsCount);
    cppNet->resizeTensor(cppTensor, cppDims);
}

/**
 * @brief resize given tensor by nchw.
 * @param net      given Interpreter.
 * @param tensor    given tensor.
 * @param batch  / N.
 * @param channel   / C.
 * @param height / H.
 * @param width / W
 */
void MNN_Interpreter_resizeTensor4D(struct MNN_Interpreter* net, struct MNN_Tensor* tensor, int batch, int channel, int height, int width) {
    auto cppNet = reinterpret_cast<Interpreter*>(net);
    auto cppTensor = reinterpret_cast<Tensor*>(tensor);
    cppNet->resizeTensor(cppTensor, batch, channel, height, width);
}

/**
 * @brief get backend used to create given tensor.
 * @param net      given Interpreter.
 * @param session   given session.
 * @param tensor    given tensor.
 * @return backend used to create given tensor, may be NULL.
 */
const struct MNN_Backend* MNN_Interpreter_getBackend(const struct MNN_Interpreter* net, const struct MNN_Session* session, const struct MNN_Tensor* tensor) {
    auto cppNet = reinterpret_cast<const Interpreter*>(net);
    auto cppSession = reinterpret_cast<const Session*>(session);
    auto cppTensor = reinterpret_cast<const Tensor*>(tensor);
    return reinterpret_cast<const struct MNN_Backend*>(cppNet->getBackend(cppSession, cppTensor));
}

/**
 * @brief get business code (model identifier).
 * @param net      given Interpreter.
 * @return business code.
 */
const char* MNN_Interpreter_bizCode(const struct MNN_Interpreter* net) {
    auto cppNet = reinterpret_cast<const Interpreter*>(net);
    return cppNet->bizCode();
}

/**
 * @brief get model UUID
 * @param net      given Interpreter.
 * @return Model UUID.
 */
const char* MNN_Interpreter_uuid(const struct MNN_Interpreter* net) {
    auto cppNet = reinterpret_cast<const Interpreter*>(net);
    return cppNet->uuid();
}


// C++适配函数：将C回调转换为C++可调用逻辑
static int wrapTensorCallback(const std::vector<MNN::Tensor*>& cppTensors, const std::string& opName, void* userData) {
    // 1. 将C++的Tensor向量转换为C的Tensor指针数组
    const MNN_Tensor** cTensors = new const MNN_Tensor*[cppTensors.size()];
    for (size_t i = 0; i < cppTensors.size(); ++i) {
        // 强制转换：C++的Tensor* -> C的MNN_Tensor*
        cTensors[i] = reinterpret_cast<const MNN_Tensor*>(cppTensors[i]);
    }

    // 2. 获取C回调函数和用户数据
    MNN_CallbackContext* ctx = static_cast<MNN_CallbackContext*>(userData);
    int result = 0;
    if (ctx->callback) {
        // 调用C的回调函数
        result = ctx->callback(cTensors, cppTensors.size(), opName.c_str(), ctx->userData);
    }

    // 3. 释放临时数组
    delete[] cTensors;
    return result;
}
MNN_ErrorCode MNN_Interpreter_runSessionWithCallBack(const MNN_Interpreter* interpreter, const MNN_Session* session,
                                                    MNN_TensorCallBack before, MNN_TensorCallBack after,
                                                    MNN_BOOL sync, void* userData){
    auto cppNet = reinterpret_cast<const Interpreter*>(interpreter);
    auto cppSession = reinterpret_cast<const Session*>(session);
    // 定义C++的回调类型（匹配MNN的TensorCallBack）
    using MNN::TensorCallBack;
    // 包装before回调
    MNN_CallbackContext beforeCtx{before, userData}; // 如果需要传递userData，需扩展接口
    TensorCallBack beforeCallback = [&](const std::vector<MNN::Tensor*>& tensors, const std::string& opName) -> int {
        return wrapTensorCallback(tensors, opName, &beforeCtx);
    };

    // 包装after回调
    MNN_CallbackContext afterCtx{after, userData};
    TensorCallBack afterCallback = [&](const std::vector<MNN::Tensor*>& tensors, const std::string& opName) -> int {
        return wrapTensorCallback(tensors, opName, &afterCtx);
    };

    // 调用MNN的C++接口（sync转换为bool）
    return static_cast<MNN_ErrorCode>(cppNet->runSessionWithCallBack(cppSession, beforeCallback, afterCallback, sync != 0));
}

static int wrapTensorCallbackWithInfo(const std::vector<Tensor*>& cpp_tensors, 
                        const OperatorInfo* cpp_info, 
                        void* userData) {
    // 1. C++ Tensor列表转C的Tensor指针数组
    size_t tensor_count = cpp_tensors.size();
    const MNN_Tensor** c_tensors = new const MNN_Tensor*[tensor_count];
    for (size_t i = 0; i < tensor_count; ++i) {
        c_tensors[i] = reinterpret_cast<const MNN_Tensor*>(cpp_tensors[i]);
    }

    // 2. C++ OperatorInfo转C的OperatorInfo指针
    const MNN_OperatorInfo* c_info = reinterpret_cast<const MNN_OperatorInfo*>(cpp_info);

    // 3. 调用C回调
    MNN_CallbackWithInfoContext* ctx = static_cast<MNN_CallbackWithInfoContext*>(userData);
    int result = 0;
    if (ctx->callback) {
        // 调用C的回调函数
        result = ctx->callback(c_tensors, tensor_count, c_info, ctx->userData);
    }

    // 4. 释放临时数组
    delete[] c_tensors;
    return result;
}
MNN_ErrorCode MNN_Interpreter_runSessionWithCallBackInfo(const MNN_Interpreter* interpreter, const MNN_Session* session,
                                                        MNN_TensorCallBackWithInfo before, MNN_TensorCallBackWithInfo after,
                                                        MNN_BOOL sync, void* userData){
    // 转换C类型到C++类型
    const Interpreter* cpp_interpreter = reinterpret_cast<const Interpreter*>(interpreter);
    const Session* cpp_session = reinterpret_cast<const Session*>(session);

    // 包装前回调：C函数指针 -> C++ std::function
    MNN_CallbackWithInfoContext beforeCtx{before, userData};
    TensorCallBackWithInfo cpp_before = [&](const std::vector<Tensor*>& tensors, const OperatorInfo* info) -> int {
        return wrapTensorCallbackWithInfo(tensors, info, &beforeCtx);
    };

    // 包装后回调：C函数指针 -> C++ std::function
    MNN_CallbackWithInfoContext afterCtx{after, userData};
    TensorCallBackWithInfo cpp_after = [&](const std::vector<Tensor*>& tensors, const OperatorInfo* info) -> int {
        return wrapTensorCallbackWithInfo(tensors, info, &afterCtx);
    };

    // 调用MNN C++接口
    ErrorCode cpp_err = cpp_interpreter->runSessionWithCallBackInfo(cpp_session, cpp_before, cpp_after, sync != 0);
    
    // 转换C++错误码到C错误码
    return static_cast<MNN_ErrorCode>(cpp_err);
}

// 转换std::map<std::string, Tensor*>到MNNNamedTensorList
static MNN_NamedTensorList convertMapToTensorList(const std::map<std::string, Tensor*>& tensorMap) {
    MNN_NamedTensorList list;
    list.count = static_cast<int>(tensorMap.size());
    list.tensors = static_cast<MNN_NamedTensor*>(malloc(list.count * sizeof(MNN_NamedTensor)));
    
    int i = 0;
    for (const auto& pair : tensorMap) {
        list.tensors[i].name = strdup(pair.first.c_str());  // 复制字符串（需手动释放）
        list.tensors[i].tensor = reinterpret_cast<MNN_Tensor*>(pair.second);
        i++;
    }
    return list;
}

MNN_NamedTensorList MNN_Interpreter_GetSessionInputAll(MNN_Interpreter* interpreter, MNN_Session* session) {
    MNN_NamedTensorList empty = {0, nullptr};
    if (!interpreter || !session) return empty;
    const auto& tensorMap = reinterpret_cast<Interpreter*>(interpreter)->getSessionInputAll(reinterpret_cast<Session*>(session));
    return convertMapToTensorList(tensorMap);
}

MNN_NamedTensorList MNN_Interpreter_GetSessionOutputAll(MNN_Interpreter* interpreter, MNN_Session* session) {
    MNN_NamedTensorList empty = {0, nullptr};
    if (!interpreter || !session) return empty;
    const auto& tensorMap = reinterpret_cast<Interpreter*>(interpreter)->getSessionOutputAll(reinterpret_cast<Session*>(session));
    return convertMapToTensorList(tensorMap);
}

void MNN_NamedTensorList_Free(MNN_NamedTensorList list) {
    if (list.tensors) {
        for (int i = 0; i < list.count; i++) {
            free(const_cast<char*>(list.tensors[i].name));  // 释放strdup的字符串
        }
        free(list.tensors);
    }
}

// 包装结构体：存储MNN::RuntimeInfo（std::pair）
struct MNN_RuntimeInfo {
    MNN::RuntimeInfo cppRuntimeInfo; // 直接存储std::pair值，内部智能指针自动管理内存
};

// Runtime management
MNN_RuntimeInfo* MNN_Interpreter_createRuntime(const MNN_ScheduleConfig* configs, int configsCount) {
    if (!configs || configsCount <= 0) return nullptr;
    
    std::vector<ScheduleConfig> cppConfigs;
    cppConfigs.reserve(configsCount);
    
    for (int i = 0; i < configsCount; i++) {
        const MNN_ScheduleConfig* config = &configs[i];
        MNN::ScheduleConfig cppConfig = convertToCppScheduleConfig(config);
        
        cppConfigs.push_back(cppConfig);
    }
    
    // 2. 调用MNN的createRuntime，获取std::pair值类型的RuntimeInfo
    MNN::RuntimeInfo cppRuntimeInfo = MNN::Interpreter::createRuntime(cppConfigs);

    // 3. 堆上分配包装结构体，存储cppRuntimeInfo（std::pair会被拷贝，内部shared_ptr引用计数+1）
    MNN_RuntimeInfo* runtimeInfoPtr = new MNN_RuntimeInfo();
    runtimeInfoPtr->cppRuntimeInfo = std::move(cppRuntimeInfo); // 用move避免深拷贝，提升性能

    return runtimeInfoPtr;
}

void MNN_RuntimeInfo_destroy(MNN_RuntimeInfo* runtime) {
    if (runtime) {
        delete runtime;
    }
}

// OperatorInfo
const char* MNN_OperatorInfo_name(const MNN_OperatorInfo* info) {
    return reinterpret_cast<const OperatorInfo*>(info)->name().c_str();
}

const char* MNN_OperatorInfo_type(const MNN_OperatorInfo* info) {
    return reinterpret_cast<const OperatorInfo*>(info)->type().c_str();
}

float MNN_OperatorInfo_flops(const MNN_OperatorInfo* info) {
    return reinterpret_cast<const OperatorInfo*>(info)->flops();
}

/*void MNN_OperatorInfo_destroy(const MNN_OperatorInfo* info) {
    if (info) {
        delete reinterpret_cast<const OperatorInfo*>(info);
    }
MNN::OperatorInfo 类的析构函数在头文件中声明为 protected
这意味着：
    只能通过类自身或其派生类调用析构函数
    外部代码（如 MNN_OperatorInfo_destroy）直接 delete 对象会触发访问权限错误    
}*/