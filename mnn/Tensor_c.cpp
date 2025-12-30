//
//  Tensor_c.cpp
//  MNN
//
//  Created by MNN on 2018/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/Tensor.hpp>
#include <Tensor_c.h>
#include <vector>

using namespace MNN;

// Tensor creation and destruction
MNN_PUBLIC struct MNN_Tensor* MNN_Tensor_Create(int dimSize, enum MNN_DimensionType type) {
    Tensor* tensor = new Tensor(dimSize, static_cast<Tensor::DimensionType>(type));
    return reinterpret_cast<struct MNN_Tensor*>(tensor);
}

MNN_PUBLIC struct MNN_Tensor* MNN_Tensor_CreateFromExisting(const struct MNN_Tensor* tensor, enum MNN_DimensionType type, MNN_BOOL allocMemory) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    Tensor* newTensor = new Tensor(cppTensor, static_cast<Tensor::DimensionType>(type), allocMemory);
    return reinterpret_cast<struct MNN_Tensor*>(newTensor);
}

MNN_PUBLIC struct MNN_Tensor* MNN_Tensor_CreateDevice(const int* shape, int shapeSize, halide_type_t type, enum MNN_DimensionType dimType) {
    std::vector<int> shapeVec(shape, shape + shapeSize);
    Tensor* tensor = Tensor::createDevice(shapeVec, type, static_cast<Tensor::DimensionType>(dimType));
    return reinterpret_cast<struct MNN_Tensor*>(tensor);
}

MNN_PUBLIC struct MNN_Tensor* MNN_Tensor_CreateHost(const int* shape, int shapeSize, halide_type_t type, void* data, enum MNN_DimensionType dimType) {
    std::vector<int> shapeVec(shape, shape + shapeSize);
    Tensor* tensor = Tensor::create(shapeVec, type, data, static_cast<Tensor::DimensionType>(dimType));
    return reinterpret_cast<struct MNN_Tensor*>(tensor);
}

MNN_PUBLIC struct MNN_Tensor* MNN_Tensor_Clone(const struct MNN_Tensor* src, MNN_BOOL deepCopy) {
    const Tensor* cppSrc = reinterpret_cast<const Tensor*>(src);
    Tensor* tensor = Tensor::clone(cppSrc, deepCopy);
    return reinterpret_cast<struct MNN_Tensor*>(tensor);
}

MNN_PUBLIC void MNN_Tensor_Destroy(struct MNN_Tensor* tensor) {
    Tensor* cppTensor = reinterpret_cast<Tensor*>(tensor);
    Tensor::destroy(cppTensor);
}

// Tensor data operations
MNN_PUBLIC MNN_BOOL MNN_Tensor_CopyFromHostTensor(struct MNN_Tensor* tensor, const struct MNN_Tensor* hostTensor) {
    Tensor* cppTensor = reinterpret_cast<Tensor*>(tensor);
    const Tensor* cppHostTensor = reinterpret_cast<const Tensor*>(hostTensor);
    return cppTensor->copyFromHostTensor(cppHostTensor);
}

MNN_PUBLIC MNN_BOOL MNN_Tensor_CopyToHostTensor(const struct MNN_Tensor* tensor, struct MNN_Tensor* hostTensor) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    Tensor* cppHostTensor = reinterpret_cast<Tensor*>(hostTensor);
    return cppTensor->copyToHostTensor(cppHostTensor);
}

MNN_PUBLIC struct MNN_Tensor* MNN_Tensor_CreateHostTensorFromDevice(const struct MNN_Tensor* deviceTensor, MNN_BOOL copyData) {
    const Tensor* cppDeviceTensor = reinterpret_cast<const Tensor*>(deviceTensor);
    Tensor* tensor = Tensor::createHostTensorFromDevice(cppDeviceTensor, copyData);
    return reinterpret_cast<struct MNN_Tensor*>(tensor);
}

// Tensor properties access
MNN_PUBLIC const halide_buffer_t* MNN_Tensor_Buffer(const struct MNN_Tensor* tensor) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    return &cppTensor->buffer();
}

MNN_PUBLIC halide_buffer_t* MNN_Tensor_MutableBuffer(struct MNN_Tensor* tensor) {
    Tensor* cppTensor = reinterpret_cast<Tensor*>(tensor);
    return &cppTensor->buffer();
}

MNN_PUBLIC enum MNN_DimensionType MNN_Tensor_GetDimensionType(const struct MNN_Tensor* tensor) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    return static_cast<enum MNN_DimensionType>(cppTensor->getDimensionType());
}

MNN_PUBLIC enum MNN_HandleDataType MNN_Tensor_GetHandleDataType(const struct MNN_Tensor* tensor) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    return static_cast<enum MNN_HandleDataType>(cppTensor->getHandleDataType());
}

MNN_PUBLIC void MNN_Tensor_SetType(struct MNN_Tensor* tensor, int type) {
    Tensor* cppTensor = reinterpret_cast<Tensor*>(tensor);
    cppTensor->setType(type);
}

/*MNN_PUBLIC halide_type_t MNN_Tensor_GetType(const struct MNN_Tensor* tensor) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    return cppTensor->getType();
}
C 语言不支持「非平凡的用户定义类型（UDT）」作为函数返回值（`halide_type_t` 是 C++ 结构体，包含对齐属性/复杂成员），而 `extern "C"` 声明的函数若返回该类型，MSVC 会报 C4190 警告（C 链接不兼容 UDT 返回值）。
*/
void MNN_Tensor_GetHalideType(const MNN_Tensor* tensor, struct halide_type_t* outType) {
    if (!outType) return; // 输出参数不能为空
    
    // 默认值：float32标量
    halide_type_t defaultType;
    defaultType.code = halide_type_float;
    defaultType.bits = 32;
    defaultType.lanes = 1;
    
    if (!tensor) {
        *outType = defaultType;
        return;
    }
    
    // 读取MNN Tensor类型
    const MNN::Tensor* cppTensor = reinterpret_cast<const MNN::Tensor*>(tensor);
    *outType = cppTensor->getType();
}

MNN_PUBLIC void* MNN_Tensor_Host(const struct MNN_Tensor* tensor) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    return cppTensor->host<uint8_t>();
}

MNN_PUBLIC uint64_t MNN_Tensor_DeviceId(const struct MNN_Tensor* tensor) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    return cppTensor->deviceId();
}

// Tensor shape and size
MNN_PUBLIC int MNN_Tensor_Dimensions(const struct MNN_Tensor* tensor) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    return cppTensor->dimensions();
}

MNN_PUBLIC int* MNN_Tensor_Shape(const struct MNN_Tensor* tensor, int* shapeSize) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    std::vector<int> shapeVec = cppTensor->shape();
    *shapeSize = static_cast<int>(shapeVec.size());
    int* shape = new int[*shapeSize];
    std::copy(shapeVec.begin(), shapeVec.end(), shape);
    return shape;
}

MNN_PUBLIC void MNN_Tensor_FreeShape(int* shape) {
    delete[] shape;
}

MNN_PUBLIC int MNN_Tensor_Size(const struct MNN_Tensor* tensor) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    return cppTensor->size();
}

MNN_PUBLIC size_t MNN_Tensor_USize(const struct MNN_Tensor* tensor) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    return cppTensor->usize();
}

MNN_PUBLIC int MNN_Tensor_ElementSize(const struct MNN_Tensor* tensor) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    return cppTensor->elementSize();
}

// Tensor dimension accessors
MNN_PUBLIC int MNN_Tensor_Width(const struct MNN_Tensor* tensor) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    return cppTensor->width();
}

MNN_PUBLIC int MNN_Tensor_Height(const struct MNN_Tensor* tensor) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    return cppTensor->height();
}

MNN_PUBLIC int MNN_Tensor_Channel(const struct MNN_Tensor* tensor) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    return cppTensor->channel();
}

MNN_PUBLIC int MNN_Tensor_Batch(const struct MNN_Tensor* tensor) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    return cppTensor->batch();
}

MNN_PUBLIC int MNN_Tensor_Stride(const struct MNN_Tensor* tensor, int index) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    return cppTensor->stride(index);
}

MNN_PUBLIC int MNN_Tensor_Length(const struct MNN_Tensor* tensor, int index) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    return cppTensor->length(index);
}

MNN_PUBLIC void MNN_Tensor_SetStride(struct MNN_Tensor* tensor, int index, int stride) {
    Tensor* cppTensor = reinterpret_cast<Tensor*>(tensor);
    cppTensor->setStride(index, stride);
}

MNN_PUBLIC void MNN_Tensor_SetLength(struct MNN_Tensor* tensor, int index, int length) {
    Tensor* cppTensor = reinterpret_cast<Tensor*>(tensor);
    cppTensor->setLength(index, length);
}

// Device operations
MNN_PUBLIC MNN_BOOL MNN_Tensor_GetDeviceInfo(const struct MNN_Tensor* tensor, void* dst, int forwardType) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    return cppTensor->getDeviceInfo(dst, forwardType);
}

// Debug operations
MNN_PUBLIC void MNN_Tensor_Print(const struct MNN_Tensor* tensor) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    cppTensor->print();
}

MNN_PUBLIC void MNN_Tensor_PrintShape(const struct MNN_Tensor* tensor) {
    const Tensor* cppTensor = reinterpret_cast<const Tensor*>(tensor);
    cppTensor->printShape();
}

// GPU operations
MNN_PUBLIC void* MNN_Tensor_Map(struct MNN_Tensor* tensor, enum MNN_MapType mtype, enum MNN_DimensionType dtype) {
    Tensor* cppTensor = reinterpret_cast<Tensor*>(tensor);
    return cppTensor->map(static_cast<Tensor::MapType>(mtype), static_cast<Tensor::DimensionType>(dtype));
}

MNN_PUBLIC void MNN_Tensor_Unmap(struct MNN_Tensor* tensor, enum MNN_MapType mtype, enum MNN_DimensionType dtype, void* mapPtr) {
    Tensor* cppTensor = reinterpret_cast<Tensor*>(tensor);
    cppTensor->unmap(static_cast<Tensor::MapType>(mtype), static_cast<Tensor::DimensionType>(dtype), mapPtr);
}

MNN_PUBLIC int MNN_Tensor_Wait(struct MNN_Tensor* tensor, enum MNN_MapType mtype, MNN_BOOL finish) {
    Tensor* cppTensor = reinterpret_cast<Tensor*>(tensor);
    return cppTensor->wait(static_cast<Tensor::MapType>(mtype), finish);
}

MNN_PUBLIC MNN_BOOL MNN_Tensor_SetDevicePtr(struct MNN_Tensor* tensor, const void* devicePtr, int memoryType) {
    Tensor* cppTensor = reinterpret_cast<Tensor*>(tensor);
    return cppTensor->setDevicePtr(devicePtr, memoryType);
}
