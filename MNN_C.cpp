#include "MNN_C.h"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>
#include <MNN/Matrix.h>
#include <vector>

using namespace MNN;

// Interpreter functions
MNN_Interpreter* MNN_Interpreter_createFromFile(const char* file) {
    return (MNN_Interpreter*)Interpreter::createFromFile(file);
}

MNN_Interpreter* MNN_Interpreter_createFromBuffer(const void* buffer, size_t size) {
    return (MNN_Interpreter*)Interpreter::createFromBuffer(buffer, size);
}

void MNN_Interpreter_destroy(MNN_Interpreter* net) {
    Interpreter::destroy((Interpreter*)net);
}

MNN_Session* MNN_Interpreter_createSession(MNN_Interpreter* net, const MNN_ScheduleConfig* config) {
    ScheduleConfig cppConfig;
    cppConfig.type = (MNNForwardType)config->datatype;
    cppConfig.numThread = config->numThread;
    cppConfig.backupType = (MNNForwardType)config->backupType;
    return (MNN_Session*)((Interpreter*)net)->createSession(cppConfig);
}

MNN_BOOL MNN_Interpreter_releaseSession(MNN_Interpreter* net, MNN_Session* session) {
    return ((Interpreter*)net)->releaseSession((Session*)session);
}

void MNN_Interpreter_resizeSession(MNN_Interpreter* net, MNN_Session* session) {
    ((Interpreter*)net)->resizeSession((Session*)session);
}

void MNN_Interpreter_resizeTensor(MNN_Interpreter* net, MNN_Tensor* tensor, const int* dims, int dimSize) {
    std::vector<int> cppDims(dims, dims + dimSize);
    ((Interpreter*)net)->resizeTensor((Tensor*)tensor, cppDims);
}

MNN_ErrorCode MNN_Interpreter_runSession(MNN_Interpreter* net, MNN_Session* session) {
    return (MNN_ErrorCode)((Interpreter*)net)->runSession((Session*)session);
}

MNN_Tensor* MNN_Interpreter_getSessionInput(MNN_Interpreter* net, MNN_Session* session, const char* name) {
    return (MNN_Tensor*)((Interpreter*)net)->getSessionInput((Session*)session, name);
}

MNN_Tensor* MNN_Interpreter_getSessionOutput(MNN_Interpreter* net, MNN_Session* session, const char* name) {
    return (MNN_Tensor*)((Interpreter*)net)->getSessionOutput((Session*)session, name);
}

// Tensor functions
MNN_Tensor* MNN_Tensor_createHostTensorFromDevice(const MNN_Tensor* deviceTensor, MNN_BOOL copyData) {
    return (MNN_Tensor*)Tensor::createHostTensorFromDevice((const Tensor*)deviceTensor, copyData);
}

MNN_Tensor* MNN_Tensor_create(const MNN_Tensor* tensor, MNN_DimensionType type, MNN_BOOL allocMemory) {
    return (MNN_Tensor*)new Tensor((const Tensor*)tensor, (Tensor::DimensionType)type, allocMemory);
}

void MNN_Tensor_destroy(MNN_Tensor* tensor) {
    delete (Tensor*)tensor;
}

MNN_BOOL MNN_Tensor_copyFromHostTensor(MNN_Tensor* deviceTensor, const MNN_Tensor* hostTensor) {
    return ((Tensor*)deviceTensor)->copyFromHostTensor((const Tensor*)hostTensor);
}

MNN_BOOL MNN_Tensor_copyToHostTensor(const MNN_Tensor* deviceTensor, MNN_Tensor* hostTensor) {
    return ((const Tensor*)deviceTensor)->copyToHostTensor((Tensor*)hostTensor);
}

float* MNN_Tensor_getFloatData(const MNN_Tensor* tensor) {
    return ((const Tensor*)tensor)->host<float>();
}

int MNN_Tensor_elementSize(const MNN_Tensor* tensor) {
    return ((const Tensor*)tensor)->elementSize();
}

void MNN_Tensor_getShape(const MNN_Tensor* tensor, int* shape, int shapeSize) {
    std::vector<int> cppShape = ((const Tensor*)tensor)->shape();
    for (int i = 0; i < shapeSize && i < (int)cppShape.size(); ++i) {
        shape[i] = cppShape[i];
    }
}

// ImageProcess wrapper struct
struct MNN_CV_ImageProcess {
    std::shared_ptr<MNN::CV::ImageProcess> imageProcess;
};

// ImageProcess functions
MNN_CV_ImageProcess* MNN_CV_ImageProcess_create(const MNN_CV_ImageProcess_Config* config, const MNN_Tensor* dstTensor) {
    MNN::CV::ImageProcess::Config cppConfig;
    if (config) {
        cppConfig.filterType = (MNN::CV::Filter)config->filterType;
        cppConfig.sourceFormat = (MNN::CV::ImageFormat)config->sourceFormat;
        cppConfig.destFormat = (MNN::CV::ImageFormat)config->destFormat;
        for (int i = 0; i < 4; ++i) {
            cppConfig.mean[i] = config->mean[i];
            cppConfig.normal[i] = config->normal[i];
        }
        cppConfig.wrap = (MNN::CV::Wrap)config->wrap;
    }
    
    auto* wrapper = new MNN_CV_ImageProcess();
    wrapper->imageProcess.reset(MNN::CV::ImageProcess::create(cppConfig, (const MNN::Tensor*)dstTensor));
    return wrapper;
}

void MNN_CV_ImageProcess_destroy(MNN_CV_ImageProcess* imageProcess) {
    if (imageProcess) {
        delete imageProcess;
    }
}

void MNN_CV_ImageProcess_setMatrix(MNN_CV_ImageProcess* imageProcess, const float* matrix) {
    if (imageProcess && imageProcess->imageProcess && matrix) {
        MNN::CV::Matrix m;
        m.setScale(matrix[0], matrix[4]);
        m.setTranslate(matrix[2], matrix[5]);
        m.postRotate(matrix[1]);
        imageProcess->imageProcess->setMatrix(m);
    }
}

MNN_ErrorCode MNN_CV_ImageProcess_convert(MNN_CV_ImageProcess* imageProcess, const uint8_t* source, int iw, int ih, int stride, MNN_Tensor* dest) {
    if (!imageProcess || !imageProcess->imageProcess || !source || !dest) {
        return INVALID_DATA;
    }
    
    auto result = imageProcess->imageProcess->convert(source, iw, ih, stride, (MNN::Tensor*)dest);
    return (MNN_ErrorCode)result;
}

void MNN_CV_ImageProcess_setPadding(MNN_CV_ImageProcess* imageProcess, uint8_t value) {
    if (imageProcess && imageProcess->imageProcess) {
        imageProcess->imageProcess->setPadding(value);
    }
}

// Utility functions
const char* MNN_getVersion() {
    return getVersion();
}
