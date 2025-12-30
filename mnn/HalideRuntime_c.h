#ifndef MNN_HALIDE_HALIDERUNTIME_C_H
#define MNN_HALIDE_HALIDERUNTIME_C_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// ====================== 编译器特性宏 ======================
#ifdef _MSC_VER
#define HALIDE_ALWAYS_INLINE __forceinline
#define HALIDE_NEVER_INLINE __declspec(noinline)
#else
#define HALIDE_ALWAYS_INLINE __attribute__((always_inline)) inline
#define HALIDE_NEVER_INLINE __attribute__((noinline))
#endif

// 内存对齐宏
#ifndef HALIDE_ATTRIBUTE_ALIGN
    #ifdef _MSC_VER
        #define HALIDE_ATTRIBUTE_ALIGN(x) __declspec(align(x))
    #else
        #define HALIDE_ATTRIBUTE_ALIGN(x) __attribute__((aligned(x)))
    #endif
#endif

// ====================== 枚举类型 ======================
/** Halide类型系统中的基础类型编码 */
typedef enum halide_type_code_t {
    halide_type_int = 0,    // 有符号整数
    halide_type_uint = 1,   // 无符号整数
    halide_type_float = 2,  // IEEE浮点数
    halide_type_handle = 3, // 不透明指针类型（void*）
    halide_type_bfloat = 4  // bfloat格式浮点数
} halide_type_code_t;

/** Buffer标志位 */
typedef enum halide_buffer_flags {
    halide_buffer_flag_host_dirty = 1,   // 主机内存数据脏
    halide_buffer_flag_device_dirty = 2  // 设备内存数据脏
} halide_buffer_flags;

// ====================== 前置声明 ======================
struct halide_buffer_t;
struct halide_device_interface_impl_t;

// ====================== 核心结构体 ======================
/** Halide类型系统的运行时标签（固定32位大小） */
struct halide_type_t {
    #ifdef _MSC_VER
    HALIDE_ATTRIBUTE_ALIGN(1) uint8_t code; // 对应halide_type_code_t
    #else
    HALIDE_ATTRIBUTE_ALIGN(1) halide_type_code_t code;
    #endif

    /** 单个标量值的精度位数 */
    HALIDE_ATTRIBUTE_ALIGN(1) uint8_t bits;

    /** 向量元素个数（标量为1） */
    HALIDE_ATTRIBUTE_ALIGN(2) uint16_t lanes;
};

/** 设备接口函数指针类型定义（简化结构体声明） */
typedef int (*halide_device_malloc_t)(void *user_context, struct halide_buffer_t *buf,
                                     const struct halide_device_interface_t *device_interface);
typedef int (*halide_device_free_t)(void *user_context, struct halide_buffer_t *buf);
typedef int (*halide_device_sync_t)(void *user_context, struct halide_buffer_t *buf);
typedef void (*halide_device_release_t)(void *user_context,
                                       const struct halide_device_interface_t *device_interface);
typedef int (*halide_copy_to_host_t)(void *user_context, struct halide_buffer_t *buf);
typedef int (*halide_copy_to_device_t)(void *user_context, struct halide_buffer_t *buf,
                                      const struct halide_device_interface_t *device_interface);
typedef int (*halide_device_and_host_malloc_t)(void *user_context, struct halide_buffer_t *buf,
                                              const struct halide_device_interface_t *device_interface);
typedef int (*halide_device_and_host_free_t)(void *user_context, struct halide_buffer_t *buf);
typedef int (*halide_buffer_copy_t)(void *user_context, struct halide_buffer_t *src,
                                   const struct halide_device_interface_t *dst_device_interface, struct halide_buffer_t *dst);
typedef int (*halide_device_crop_t)(void *user_context, const struct halide_buffer_t *src,
                                   struct halide_buffer_t *dst);
typedef int (*halide_device_release_crop_t)(void *user_context, struct halide_buffer_t *buf);
typedef int (*halide_wrap_native_t)(void *user_context, struct halide_buffer_t *buf, uint64_t handle,
                                   const struct halide_device_interface_t *device_interface);
typedef int (*halide_detach_native_t)(void *user_context, struct halide_buffer_t *buf);

/** GPU API设备接口（管理设备内存分配/同步等） */
struct halide_device_interface_t {
    halide_device_malloc_t device_malloc;
    halide_device_free_t device_free;
    halide_device_sync_t device_sync;
    halide_device_release_t device_release;
    halide_copy_to_host_t copy_to_host;
    halide_copy_to_device_t copy_to_device;
    halide_device_and_host_malloc_t device_and_host_malloc;
    halide_device_and_host_free_t device_and_host_free;
    halide_buffer_copy_t buffer_copy;
    halide_device_crop_t device_crop;
    halide_device_release_crop_t device_release_crop;
    halide_wrap_native_t wrap_native;
    halide_detach_native_t detach_native;
    const struct halide_device_interface_impl_t *impl; // 设备API具体实现（不透明）
};

/** 维度信息结构体 */
struct halide_dimension_t {
    int32_t min;     // 维度最小值
    int32_t extent;  // 维度长度
    int32_t stride;  // 维度步长
    uint32_t flags;  // 维度标志（预留）
};

/** Halide Buffer的原始表示（对应Halide::Buffer<T>的C接口） */
struct halide_buffer_t {
    uint64_t device;                      // GPU设备内存句柄
    const struct halide_device_interface_t *device_interface; // 设备接口
    uint8_t* host;                        // 主机内存起始地址
    uint64_t flags;                       // 标志位（halide_buffer_flags）
    struct halide_type_t type;            // 元素类型
    int32_t dimensions;                   // 维度数
    struct halide_dimension_t *dim;       // 维度信息数组（用户管理内存）
    void *padding;                        // 8字节对齐填充
};

// ====================== 辅助函数（替代C++模板） ======================
/**
 * 获取基础类型对应的halide_type_t（替代C++模板halide_type_of）
 * @param code 类型编码（halide_type_code_t）
 * @param bits 精度位数
 * @param lanes 向量元素数（默认1）
 * @return halide_type_t结构体
 */
HALIDE_ALWAYS_INLINE struct halide_type_t halide_type_create(halide_type_code_t code, uint8_t bits, uint16_t lanes) {
    struct halide_type_t t;
    t.code = code;
    t.bits = bits;
    t.lanes = lanes;
    return t;
}

/** 便捷函数：创建标量类型 */
#define HALIDE_TYPE_CREATE_SCALAR(code, bits) halide_type_create(code, bits, 1)

// 常用类型的便捷创建宏（替代C++模板特化）
#define HALIDE_TYPE_FLOAT32()  HALIDE_TYPE_CREATE_SCALAR(halide_type_float, 32)
#define HALIDE_TYPE_FLOAT64()  HALIDE_TYPE_CREATE_SCALAR(halide_type_float, 64)
#define HALIDE_TYPE_BOOL()     HALIDE_TYPE_CREATE_SCALAR(halide_type_uint, 1)
#define HALIDE_TYPE_UINT8()    HALIDE_TYPE_CREATE_SCALAR(halide_type_uint, 8)
#define HALIDE_TYPE_UINT16()   HALIDE_TYPE_CREATE_SCALAR(halide_type_uint, 16)
#define HALIDE_TYPE_UINT32()   HALIDE_TYPE_CREATE_SCALAR(halide_type_uint, 32)
#define HALIDE_TYPE_UINT64()   HALIDE_TYPE_CREATE_SCALAR(halide_type_uint, 64)
#define HALIDE_TYPE_INT8()     HALIDE_TYPE_CREATE_SCALAR(halide_type_int, 8)
#define HALIDE_TYPE_INT16()    HALIDE_TYPE_CREATE_SCALAR(halide_type_int, 16)
#define HALIDE_TYPE_INT32()    HALIDE_TYPE_CREATE_SCALAR(halide_type_int, 32)
#define HALIDE_TYPE_INT64()    HALIDE_TYPE_CREATE_SCALAR(halide_type_int, 64)
#define HALIDE_TYPE_HANDLE()   HALIDE_TYPE_CREATE_SCALAR(halide_type_handle, 64)
#define HALIDE_TYPE_BFLOAT()   HALIDE_TYPE_CREATE_SCALAR(halide_type_bfloat, 16)

/**
 * 计算类型的字节大小（替代C++成员函数bytes()）
 * @param t halide_type_t结构体
 * @return 单个元素的字节数
 */
HALIDE_ALWAYS_INLINE int halide_type_bytes(const struct halide_type_t *t) {
    return (t->bits + 7) / 8;
}

/**
 * 比较两个halide_type_t是否相等（替代C++ operator==）
 * @param a 第一个类型
 * @param b 第二个类型
 * @return 相等返回true，否则false
 */
HALIDE_ALWAYS_INLINE bool halide_type_equal(const struct halide_type_t *a, const struct halide_type_t *b) {
    return (a->code == b->code) && (a->bits == b->bits) && (a->lanes == b->lanes);
}

/**
 * 比较两个halide_dimension_t是否相等（替代C++ operator==）
 * @param a 第一个维度
 * @param b 第二个维度
 * @return 相等返回true，否则false
 */
HALIDE_ALWAYS_INLINE bool halide_dimension_equal(const struct halide_dimension_t *a, const struct halide_dimension_t *b) {
    return (a->min == b->min) && (a->extent == b->extent) && (a->stride == b->stride) && (a->flags == b->flags);
}

#endif // MNN_HALIDE_HALIDERUNTIME_C_H