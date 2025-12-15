#include "MNN_C.h"
#include "MNN_Express.h"
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <vector>

using namespace MNN;
using namespace MNN::Express;

// Express module struct wrappers
struct MNN_Express_VARP {
    VARP var;
};

struct MNN_Express_Module {
    std::shared_ptr<Module> module;
};

struct MNN_Express_RuntimeManager {
    std::shared_ptr<Executor::RuntimeManager> runtimeManager;
};

// Express VARP functions
MNN_Express_VARP* MNN_Express_VARP_createConstFloat(const float* data, const int* dims, int dimSize) {
    if (!data || !dims || dimSize <= 0) {
        return nullptr;
    }
    
    // Calculate total number of elements
    size_t totalElements = 1;
    for (int i = 0; i < dimSize; ++i) {
        totalElements *= dims[i];
    }
    
    std::vector<float> cppData(data, data + totalElements);
    std::vector<int> cppDims(dims, dims + dimSize);
    
    auto* var = new MNN_Express_VARP();
    var->var = MNN::Express::_Const(cppData.data(), cppDims, MNN::Express::NHWC, halide_type_of<float>());
    return var;
}

MNN_Express_VARP* MNN_Express_VARP_createConstInt(const int* data, const int* dims, int dimSize) {
    if (!data || !dims || dimSize <= 0) {
        return nullptr;
    }
    
    // Calculate total number of elements
    size_t totalElements = 1;
    for (int i = 0; i < dimSize; ++i) {
        totalElements *= dims[i];
    }
    
    std::vector<int> cppData(data, data + totalElements);
    std::vector<int> cppDims(dims, dims + dimSize);
    
    auto* var = new MNN_Express_VARP();
    var->var = MNN::Express::_Const(cppData.data(), cppDims, MNN::Express::NHWC, halide_type_of<int>());
    return var;
}

void MNN_Express_VARP_destroy(MNN_Express_VARP* var) {
    delete var;
}

float* MNN_Express_VARP_getFloatData(MNN_Express_VARP* var) {
    if (!var || !var->var.get()) return nullptr;
    // 使用const_cast来处理const float*到float*的转换
    return const_cast<float*>(var->var->readMap<float>());
}

int MNN_Express_VARP_elementSize(MNN_Express_VARP* var) {
    if (!var || !var->var.get()) return 0;
    auto info = var->var->getInfo();
    if (!info) return 0;
    return (int)info->size;
}

void MNN_Express_VARP_getShape(MNN_Express_VARP* var, int* shape, int shapeSize) {
    if (!var || !var->var.get()) return;
    auto info = var->var->getInfo();
    if (!info) return;
    
    for (int i = 0; i < shapeSize && i < (int)info->dim.size(); ++i) {
        shape[i] = info->dim[i];
    }
}

// Express Module functions
MNN_Express_Module* MNN_Express_Module_loadFromFile(
    const char** inputs,
    int inputCount,
    const char** outputs,
    int outputCount,
    const char* fileName,
    MNN_Express_RuntimeManager* runtimeManager,
    const MNN_Express_Config* config) {
    
    std::vector<std::string> cppInputs(inputs, inputs + inputCount);
    std::vector<std::string> cppOutputs(outputs, outputs + outputCount);
    
    Module::Config cppConfig;
    if (config) {
        cppConfig.dynamic = config->dynamic;
        cppConfig.shapeMutable = config->shapeMutable;
        cppConfig.rearrange = config->rearrange;
        if (config->base) {
            cppConfig.base = config->base->module.get();
        }
    }
    
    auto* moduleWrapper = new MNN_Express_Module();
    
    if (runtimeManager) {
        moduleWrapper->module = std::shared_ptr<Module>(
            Module::load(cppInputs, cppOutputs, fileName, runtimeManager->runtimeManager, &cppConfig));
    } else {
        moduleWrapper->module = std::shared_ptr<Module>(
            Module::load(cppInputs, cppOutputs, fileName, &cppConfig));
    }
    
    return moduleWrapper;
}

void MNN_Express_Module_destroy(MNN_Express_Module* module) {
    delete module;
}

MNN_Express_VARP** MNN_Express_Module_onForward(
    MNN_Express_Module* module,
    MNN_Express_VARP** inputs,
    int inputCount,
    int* outputCount) {
    
    if (!module || !module->module) {
        *outputCount = 0;
        return nullptr;
    }
    
    std::vector<VARP> cppInputs;
    for (int i = 0; i < inputCount; ++i) {
        if (inputs[i]) {
            cppInputs.push_back(inputs[i]->var);
        }
    }
    
    std::vector<VARP> cppOutputs = module->module->onForward(cppInputs);
    
    *outputCount = (int)cppOutputs.size();
    if (*outputCount == 0) {
        return nullptr;
    }
    
    MNN_Express_VARP** outputs = (MNN_Express_VARP**)malloc(sizeof(MNN_Express_VARP*) * *outputCount);
    for (int i = 0; i < *outputCount; ++i) {
        outputs[i] = new MNN_Express_VARP();
        outputs[i]->var = cppOutputs[i];
    }
    
    return outputs;
}

// Express RuntimeManager functions
MNN_Express_RuntimeManager* MNN_Express_RuntimeManager_create(MNN_C_ForwardType type, int numThread) {
    ScheduleConfig config;
    config.type = (MNNForwardType)type;
    config.numThread = numThread;
    
    auto* runtimeWrapper = new MNN_Express_RuntimeManager();
    runtimeWrapper->runtimeManager.reset(Executor::RuntimeManager::createRuntimeManager(config));
    
    return runtimeWrapper;
}

void MNN_Express_RuntimeManager_destroy(MNN_Express_RuntimeManager* runtimeManager) {
    delete runtimeManager;
}

void MNN_Express_RuntimeManager_setHint(MNN_Express_RuntimeManager* runtimeManager, int hint, int value) {
    // 暂时不实现setHint，因为Interpreter::Hint不存在
    (void)runtimeManager;
    (void)hint;
    (void)value;
}

// Helper functions for array management
void MNN_Express_freeVARPArray(MNN_Express_VARP** array, int count) {
    if (!array) return;
    
    for (int i = 0; i < count; ++i) {
        if (array[i]) {
            MNN_Express_VARP_destroy(array[i]);
        }
    }
    
    free(array);
}

// Express MathOp functions - Binary operations
MNN_Express_VARP* MNN_Express_add(MNN_Express_VARP* x, MNN_Express_VARP* y) {
    if (!x || !y) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Add(x->var, y->var);
    return var;
}

MNN_Express_VARP* MNN_Express_subtract(MNN_Express_VARP* x, MNN_Express_VARP* y) {
    if (!x || !y) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Subtract(x->var, y->var);
    return var;
}

MNN_Express_VARP* MNN_Express_multiply(MNN_Express_VARP* x, MNN_Express_VARP* y) {
    if (!x || !y) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Multiply(x->var, y->var);
    return var;
}

MNN_Express_VARP* MNN_Express_divide(MNN_Express_VARP* x, MNN_Express_VARP* y) {
    if (!x || !y) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Divide(x->var, y->var);
    return var;
}

MNN_Express_VARP* MNN_Express_pow(MNN_Express_VARP* x, MNN_Express_VARP* y) {
    if (!x || !y) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Pow(x->var, y->var);
    return var;
}

MNN_Express_VARP* MNN_Express_minimum(MNN_Express_VARP* x, MNN_Express_VARP* y) {
    if (!x || !y) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Minimum(x->var, y->var);
    return var;
}

MNN_Express_VARP* MNN_Express_maximum(MNN_Express_VARP* x, MNN_Express_VARP* y) {
    if (!x || !y) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Maximum(x->var, y->var);
    return var;
}

MNN_Express_VARP* MNN_Express_biasAdd(MNN_Express_VARP* value, MNN_Express_VARP* bias) {
    if (!value || !bias) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _BiasAdd(value->var, bias->var);
    return var;
}

MNN_Express_VARP* MNN_Express_greater(MNN_Express_VARP* x, MNN_Express_VARP* y) {
    if (!x || !y) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Greater(x->var, y->var);
    return var;
}

MNN_Express_VARP* MNN_Express_greaterEqual(MNN_Express_VARP* x, MNN_Express_VARP* y) {
    if (!x || !y) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _GreaterEqual(x->var, y->var);
    return var;
}

MNN_Express_VARP* MNN_Express_less(MNN_Express_VARP* x, MNN_Express_VARP* y) {
    if (!x || !y) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Less(x->var, y->var);
    return var;
}

MNN_Express_VARP* MNN_Express_floorDiv(MNN_Express_VARP* x, MNN_Express_VARP* y) {
    if (!x || !y) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _FloorDiv(x->var, y->var);
    return var;
}

MNN_Express_VARP* MNN_Express_squaredDifference(MNN_Express_VARP* x, MNN_Express_VARP* y) {
    if (!x || !y) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _SquaredDifference(x->var, y->var);
    return var;
}

MNN_Express_VARP* MNN_Express_equal(MNN_Express_VARP* x, MNN_Express_VARP* y) {
    if (!x || !y) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Equal(x->var, y->var);
    return var;
}

MNN_Express_VARP* MNN_Express_lessEqual(MNN_Express_VARP* x, MNN_Express_VARP* y) {
    if (!x || !y) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _LessEqual(x->var, y->var);
    return var;
}

MNN_Express_VARP* MNN_Express_floorMod(MNN_Express_VARP* x, MNN_Express_VARP* y) {
    if (!x || !y) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _FloorMod(x->var, y->var);
    return var;
}

MNN_Express_VARP* MNN_Express_atan2(MNN_Express_VARP* x, MNN_Express_VARP* y) {
    if (!x || !y) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Atan2(x->var, y->var);
    return var;
}

// Express MathOp functions - Unary operations
MNN_Express_VARP* MNN_Express_sign(MNN_Express_VARP* a) {
    if (!a) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Sign(a->var);
    return var;
}

MNN_Express_VARP* MNN_Express_abs(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Abs(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_negative(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Negative(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_floor(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Floor(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_round(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Round(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_ceil(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Ceil(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_square(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Square(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_sqrt(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Sqrt(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_rsqrt(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Rsqrt(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_exp(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Exp(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_log(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Log(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_sin(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Sin(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_cos(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Cos(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_tan(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Tan(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_asin(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Asin(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_acos(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Acos(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_atan(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Atan(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_reciprocal(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Reciprocal(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_tanh(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Tanh(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_sigmoid(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Sigmoid(x->var);
    return var;
}

// Express MathOp functions - Reduce operations
MNN_Express_VARP* MNN_Express_reduceSum(MNN_Express_VARP* input, const int* axis, int axisSize, MNN_BOOL keepDims) {
    if (!input) return nullptr;
    
    INTS cppAxis;
    if (axis && axisSize > 0) {
        cppAxis = INTS(axis, axis + axisSize);
    }
    
    auto* var = new MNN_Express_VARP();
    var->var = _ReduceSum(input->var, cppAxis, keepDims);
    return var;
}

MNN_Express_VARP* MNN_Express_reduceMean(MNN_Express_VARP* input, const int* axis, int axisSize, MNN_BOOL keepDims) {
    if (!input) return nullptr;
    
    INTS cppAxis;
    if (axis && axisSize > 0) {
        cppAxis = INTS(axis, axis + axisSize);
    }
    
    auto* var = new MNN_Express_VARP();
    var->var = _ReduceMean(input->var, cppAxis, keepDims);
    return var;
}

MNN_Express_VARP* MNN_Express_reduceMax(MNN_Express_VARP* input, const int* axis, int axisSize, MNN_BOOL keepDims) {
    if (!input) return nullptr;
    
    INTS cppAxis;
    if (axis && axisSize > 0) {
        cppAxis = INTS(axis, axis + axisSize);
    }
    
    auto* var = new MNN_Express_VARP();
    var->var = _ReduceMax(input->var, cppAxis, keepDims);
    return var;
}

MNN_Express_VARP* MNN_Express_reduceMin(MNN_Express_VARP* input, const int* axis, int axisSize, MNN_BOOL keepDims) {
    if (!input) return nullptr;
    
    INTS cppAxis;
    if (axis && axisSize > 0) {
        cppAxis = INTS(axis, axis + axisSize);
    }
    
    auto* var = new MNN_Express_VARP();
    var->var = _ReduceMin(input->var, cppAxis, keepDims);
    return var;
}

MNN_Express_VARP* MNN_Express_reduceProd(MNN_Express_VARP* input, const int* axis, int axisSize, MNN_BOOL keepDims) {
    if (!input) return nullptr;
    
    INTS cppAxis;
    if (axis && axisSize > 0) {
        cppAxis = INTS(axis, axis + axisSize);
    }
    
    auto* var = new MNN_Express_VARP();
    var->var = _ReduceProd(input->var, cppAxis, keepDims);
    return var;
}

// Express MathOp functions - Other operations
MNN_Express_VARP* MNN_Express_castToFloat(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Cast<float>(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_castToInt(MNN_Express_VARP* x) {
    if (!x) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _Cast<int>(x->var);
    return var;
}

MNN_Express_VARP* MNN_Express_matMul(MNN_Express_VARP* a, MNN_Express_VARP* b, MNN_BOOL transposeA, MNN_BOOL transposeB) {
    if (!a || !b) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _MatMul(a->var, b->var, transposeA, transposeB);
    return var;
}

MNN_Express_VARP* MNN_Express_argMax(MNN_Express_VARP* input, int axis) {
    if (!input) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _ArgMax(input->var, axis);
    return var;
}

MNN_Express_VARP* MNN_Express_argMin(MNN_Express_VARP* input, int axis) {
    if (!input) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _ArgMin(input->var, axis);
    return var;
}

MNN_Express_VARP* MNN_Express_batchMatMul(MNN_Express_VARP* x, MNN_Express_VARP* y, MNN_BOOL adj_x, MNN_BOOL adj_y) {
    if (!x || !y) return nullptr;
    auto* var = new MNN_Express_VARP();
    var->var = _BatchMatMul(x->var, y->var, adj_x, adj_y);
    return var;
}