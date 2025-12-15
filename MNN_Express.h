#ifndef MNN_Express_H
#define MNN_Express_H

#include <stddef.h>
#include <stdint.h>

// �����꣺Windows�µ���������ƽ̨����
#ifdef MNN_C_EXPORTS
    // �������� libmnn.dll ʱ���� MNN_C_EXPORTS����ʱ�� dllexport
    #define MNN_C_API __declspec(dllexport)
#else
    #define MNN_C_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Express module forward declarations
typedef struct MNN_Express_VARP MNN_Express_VARP;
typedef struct MNN_Express_Module MNN_Express_Module;
typedef struct MNN_Express_Config MNN_Express_Config;
typedef struct MNN_Express_RuntimeManager MNN_Express_RuntimeManager;

// Express Config
typedef struct MNN_Express_Config {
    MNN_BOOL dynamic;
    MNN_BOOL shapeMutable;
    MNN_BOOL rearrange;
    const MNN_Express_Module* base;
}
 MNN_Express_Config;

// Express VARP functions
MNN_C_API MNN_Express_VARP* MNN_Express_VARP_createConstFloat(const float* data, const int* dims, int dimSize);
MNN_C_API MNN_Express_VARP* MNN_Express_VARP_createConstInt(const int* data, const int* dims, int dimSize);
MNN_C_API void MNN_Express_VARP_destroy(MNN_Express_VARP* var);
MNN_C_API float* MNN_Express_VARP_getFloatData(MNN_Express_VARP* var);
MNN_C_API int MNN_Express_VARP_elementSize(MNN_Express_VARP* var);
MNN_C_API void MNN_Express_VARP_getShape(MNN_Express_VARP* var, int* shape, int shapeSize);

// Express Module functions
MNN_C_API MNN_Express_Module* MNN_Express_Module_loadFromFile(
    const char** inputs,
    int inputCount,
    const char** outputs,
    int outputCount,
    const char* fileName,
    MNN_Express_RuntimeManager* runtimeManager,
    const MNN_Express_Config* config);

MNN_C_API void MNN_Express_Module_destroy(MNN_Express_Module* module);
MNN_C_API MNN_Express_VARP** MNN_Express_Module_onForward(
    MNN_Express_Module* module,
    MNN_Express_VARP** inputs,
    int inputCount,
    int* outputCount);

// Express RuntimeManager functions
MNN_C_API MNN_Express_RuntimeManager* MNN_Express_RuntimeManager_create(MNN_C_ForwardType type, int numThread);
MNN_C_API void MNN_Express_RuntimeManager_destroy(MNN_Express_RuntimeManager* runtimeManager);
MNN_C_API void MNN_Express_RuntimeManager_setHint(MNN_Express_RuntimeManager* runtimeManager, int hint, int value);

// Helper functions for array management
MNN_C_API void MNN_Express_freeVARPArray(MNN_Express_VARP** array, int count);

// Express MathOp functions - Binary operations
MNN_C_API MNN_Express_VARP* MNN_Express_add(MNN_Express_VARP* x, MNN_Express_VARP* y);
MNN_C_API MNN_Express_VARP* MNN_Express_subtract(MNN_Express_VARP* x, MNN_Express_VARP* y);
MNN_C_API MNN_Express_VARP* MNN_Express_multiply(MNN_Express_VARP* x, MNN_Express_VARP* y);
MNN_C_API MNN_Express_VARP* MNN_Express_divide(MNN_Express_VARP* x, MNN_Express_VARP* y);
MNN_C_API MNN_Express_VARP* MNN_Express_pow(MNN_Express_VARP* x, MNN_Express_VARP* y);
MNN_C_API MNN_Express_VARP* MNN_Express_minimum(MNN_Express_VARP* x, MNN_Express_VARP* y);
MNN_C_API MNN_Express_VARP* MNN_Express_maximum(MNN_Express_VARP* x, MNN_Express_VARP* y);
MNN_C_API MNN_Express_VARP* MNN_Express_biasAdd(MNN_Express_VARP* value, MNN_Express_VARP* bias);
MNN_C_API MNN_Express_VARP* MNN_Express_greater(MNN_Express_VARP* x, MNN_Express_VARP* y);
MNN_C_API MNN_Express_VARP* MNN_Express_greaterEqual(MNN_Express_VARP* x, MNN_Express_VARP* y);
MNN_C_API MNN_Express_VARP* MNN_Express_less(MNN_Express_VARP* x, MNN_Express_VARP* y);
MNN_C_API MNN_Express_VARP* MNN_Express_floorDiv(MNN_Express_VARP* x, MNN_Express_VARP* y);
MNN_C_API MNN_Express_VARP* MNN_Express_squaredDifference(MNN_Express_VARP* x, MNN_Express_VARP* y);
MNN_C_API MNN_Express_VARP* MNN_Express_equal(MNN_Express_VARP* x, MNN_Express_VARP* y);
MNN_C_API MNN_Express_VARP* MNN_Express_lessEqual(MNN_Express_VARP* x, MNN_Express_VARP* y);
MNN_C_API MNN_Express_VARP* MNN_Express_floorMod(MNN_Express_VARP* x, MNN_Express_VARP* y);
MNN_C_API MNN_Express_VARP* MNN_Express_atan2(MNN_Express_VARP* x, MNN_Express_VARP* y);

// Express MathOp functions - Unary operations
MNN_C_API MNN_Express_VARP* MNN_Express_sign(MNN_Express_VARP* a);
MNN_C_API MNN_Express_VARP* MNN_Express_abs(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_negative(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_floor(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_round(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_ceil(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_square(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_sqrt(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_rsqrt(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_exp(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_log(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_sin(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_cos(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_tan(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_asin(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_acos(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_atan(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_reciprocal(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_tanh(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_sigmoid(MNN_Express_VARP* x);

// Express MathOp functions - Reduce operations
MNN_C_API MNN_Express_VARP* MNN_Express_reduceSum(MNN_Express_VARP* input, const int* axis, int axisSize, MNN_BOOL keepDims);
MNN_C_API MNN_Express_VARP* MNN_Express_reduceMean(MNN_Express_VARP* input, const int* axis, int axisSize, MNN_BOOL keepDims);
MNN_C_API MNN_Express_VARP* MNN_Express_reduceMax(MNN_Express_VARP* input, const int* axis, int axisSize, MNN_BOOL keepDims);
MNN_C_API MNN_Express_VARP* MNN_Express_reduceMin(MNN_Express_VARP* input, const int* axis, int axisSize, MNN_BOOL keepDims);
MNN_C_API MNN_Express_VARP* MNN_Express_reduceProd(MNN_Express_VARP* input, const int* axis, int axisSize, MNN_BOOL keepDims);

// Express MathOp functions - Other operations
MNN_C_API MNN_Express_VARP* MNN_Express_castToFloat(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_castToInt(MNN_Express_VARP* x);
MNN_C_API MNN_Express_VARP* MNN_Express_matMul(MNN_Express_VARP* a, MNN_Express_VARP* b, MNN_BOOL transposeA, MNN_BOOL transposeB);
MNN_C_API MNN_Express_VARP* MNN_Express_argMax(MNN_Express_VARP* input, int axis);
MNN_C_API MNN_Express_VARP* MNN_Express_argMin(MNN_Express_VARP* input, int axis);
MNN_C_API MNN_Express_VARP* MNN_Express_batchMatMul(MNN_Express_VARP* x, MNN_Express_VARP* y, MNN_BOOL adj_x, MNN_BOOL adj_y);

#ifdef __cplusplus
}
#endif

#endif // MNN_C_H