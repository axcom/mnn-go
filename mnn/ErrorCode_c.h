//
//  ErrorCode.h
//  MNN
//
//  Created by MNN on 2018/09/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_ErrorCode_c_h
#define MNN_ErrorCode_c_h

typedef enum {
#ifdef NO_ERROR
#undef NO_ERROR
#endif // NO_ERROR
    MNN_NO_ERROR           = 0,
    MNN_OUT_OF_MEMORY      = 1,
    MNN_NOT_SUPPORT        = 2,
    MNN_COMPUTE_SIZE_ERROR = 3,
    MNN_NO_EXECUTION       = 4,
    MNN_INVALID_VALUE      = 5,

    // User error
    MNN_INPUT_DATA_ERROR = 10,
    MNN_CALL_BACK_STOP   = 11,

    // Op Resize Error
    MNN_TENSOR_NOT_SUPPORT = 20,
    MNN_TENSOR_NEED_DIVIDE = 21,

    // File error
    MNN_FILE_CREATE_FAILED = 30,
    MNN_FILE_REMOVE_FAILED = 31,
    MNN_FILE_OPEN_FAILED   = 32,
    MNN_FILE_CLOSE_FAILED  = 33,
    MNN_FILE_RESIZE_FAILED = 34,
    MNN_FILE_SEEK_FAILED   = 35,
    MNN_FILE_NOT_EXIST     = 36,
    MNN_FILE_UNMAP_FAILED  = 37
} MNN_ErrorCode;

#endif /* MNN_ErrorCode_c_h */