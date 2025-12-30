package mnn

/*
#include "ImageProcess_c.h"
#include "Tensor_c.h"
#include "Matrix_c.h"
#include "ErrorCode_c.h"
typedef struct halide_type_t halide_type_t;
*/
import "C"
import (
	"unsafe"
)

// ImageFormat represents the image format enum
const (
	RGBA     ImageFormat = C.MNN_RGBA
	RGB      ImageFormat = C.MNN_RGB
	BGR      ImageFormat = C.MNN_BGR
	GRAY     ImageFormat = C.MNN_GRAY
	BGRA     ImageFormat = C.MNN_BGRA
	YCrCb    ImageFormat = C.MNN_YCrCb
	YUV      ImageFormat = C.MNN_YUV
	HSV      ImageFormat = C.MNN_HSV
	XYZ      ImageFormat = C.MNN_XYZ
	BGR555   ImageFormat = C.MNN_BGR555
	BGR565   ImageFormat = C.MNN_BGR565
	YUV_NV21 ImageFormat = C.MNN_YUV_NV21
	YUV_NV12 ImageFormat = C.MNN_YUV_NV12
	YUV_I420 ImageFormat = C.MNN_YUV_I420
	HSV_FULL ImageFormat = C.MNN_HSV_FULL
)

// ImageFormat is the image format type
type ImageFormat C.enum_MNN_ImageFormat

// Filter represents the filter type enum
const (
	NEAREST  Filter = C.MNN_NEAREST
	BILINEAR Filter = C.MNN_BILINEAR
	BICUBIC  Filter = C.MNN_BICUBIC
)

// Filter is the filter type
type Filter C.enum_MNN_Filter

// Wrap represents the wrap mode enum
const (
	CLAMP_TO_EDGE Wrap = C.MNN_CLAMP_TO_EDGE
	ZERO          Wrap = C.MNN_ZERO
	REPEAT        Wrap = C.MNN_REPEAT
)

// Wrap is the wrap mode type
type Wrap C.enum_MNN_Wrap

// ImageProcessConfig holds the configuration for ImageProcess
type ImageProcessConfig struct {
	FilterType   Filter
	SourceFormat ImageFormat
	DestFormat   ImageFormat
	Mean         [4]float32
	Normal       [4]float32
	Wrap         Wrap
}

// ImageProcess wraps the C MNN_ImageProcess struct
type ImageProcess struct {
	c *C.struct_MNN_ImageProcess
}

// ToC converts Go ImageProcessConfig to C MNN_ImageProcess_Config
func (config *ImageProcessConfig) ToC() *C.struct_MNN_ImageProcess_Config {
	cConfig := &C.struct_MNN_ImageProcess_Config{
		filterType:   C.enum_MNN_Filter(config.FilterType),
		sourceFormat: C.enum_MNN_ImageFormat(config.SourceFormat),
		destFormat:   C.enum_MNN_ImageFormat(config.DestFormat),
		wrap:         C.enum_MNN_Wrap(config.Wrap),
	}
	for i := 0; i < 4; i++ {
		cConfig.mean[i] = C.float(config.Mean[i])
		cConfig.normal[i] = C.float(config.Normal[i])
	}
	return cConfig
}

// DefaultImageProcessConfig returns the default configuration
func DefaultImageProcessConfig() ImageProcessConfig {
	return ImageProcessConfig{
		FilterType:   NEAREST,
		SourceFormat: RGBA,
		DestFormat:   RGBA,
		Mean:         [4]float32{0.0, 0.0, 0.0, 0.0},
		Normal:       [4]float32{1.0, 1.0, 1.0, 1.0},
		Wrap:         CLAMP_TO_EDGE,
	}
}

// CreateImageProcess creates a new ImageProcess with the given configuration and destination tensor
func CreateImageProcess(config *ImageProcessConfig, dstTensor *Tensor) *ImageProcess {
	var cConfig *C.struct_MNN_ImageProcess_Config
	if config != nil {
		cConfig = config.ToC()
	}

	var cDstTensor *C.struct_MNN_Tensor
	if dstTensor != nil {
		cDstTensor = dstTensor.c
	}

	cProcess := C.MNN_ImageProcess_create(cConfig, cDstTensor)
	return &ImageProcess{c: cProcess}
}

// CreateImageProcessV2 creates a new ImageProcess with the given parameters and destination tensor
func CreateImageProcessV2(sourceFormat, destFormat ImageFormat, means []float32, normals []float32, dstTensor *Tensor) *ImageProcess {
	var cMeans *C.float
	meanCount := 0
	if len(means) > 0 {
		cMeans = (*C.float)(unsafe.Pointer(&means[0]))
		meanCount = len(means)
	}

	var cNormals *C.float
	normalCount := 0
	if len(normals) > 0 {
		cNormals = (*C.float)(unsafe.Pointer(&normals[0]))
		normalCount = len(normals)
	}

	var cDstTensor *C.struct_MNN_Tensor
	if dstTensor != nil {
		cDstTensor = dstTensor.c
	}

	cProcess := C.MNN_ImageProcess_create_v2(
		C.enum_MNN_ImageFormat(sourceFormat),
		C.enum_MNN_ImageFormat(destFormat),
		cMeans,
		C.int(meanCount),
		cNormals,
		C.int(normalCount),
		cDstTensor,
	)
	return &ImageProcess{c: cProcess}
}

// Destroy destroys the ImageProcess instance
func (process *ImageProcess) Close() {
	if process != nil && process.c != nil {
		C.MNN_ImageProcess_destroy(process.c)
		process.c = nil
	}
}

// SetMatrix sets the affine transform matrix
func (process *ImageProcess) SetMatrix(matrix *Matrix) {
	if process != nil && process.c != nil && matrix != nil {
		C.MNN_ImageProcess_setMatrix(process.c, matrix.UnsafeC())
	}
}

// Convert converts the source image data to the destination tensor
func (process *ImageProcess) Convert(source []byte, iw, ih, stride int, dest *Tensor) ErrorCode {
	cSource := (*C.uint8_t)(unsafe.Pointer(&source[0]))

	result := C.MNN_ImageProcess_convert(
		process.c,
		cSource,
		C.int(iw),
		C.int(ih),
		C.int(stride),
		dest.c,
	)

	return ErrorCode(result)
}

// ConvertV2 converts the source image data to the destination buffer
func (process *ImageProcess) ConvertV2(source []byte, iw, ih, stride int, dest []byte, ow, oh, outputBpp, outputStride int, dataType HalideType) ErrorCode {
	cSource := (*C.uint8_t)(unsafe.Pointer(&source[0]))
	cDest := unsafe.Pointer(&dest[0])
	cType := C.halide_type_t{
		code:  C.halide_type_code_t(C.uint8_t(dataType.Code)), //C.uint8_t(dataType >> 24 & 0xFF),
		bits:  C.uint8_t(dataType.Bits),                       //C.uint8_t(dataType >> 16 & 0xFF),
		lanes: C.uint16_t(dataType.Lanes),                     //C.uint16_t(dataType & 0xFFFF),
	}

	result := C.MNN_ImageProcess_convert_v2(
		process.c,
		cSource,
		C.int(iw),
		C.int(ih),
		C.int(stride),
		cDest,
		C.int(ow),
		C.int(oh),
		C.int(outputBpp),
		C.int(outputStride),
		cType,
	)

	return ErrorCode(result)
}

// CreateImageTensor creates a new tensor for image data
func CreateImageTensor(dataType HalideType, w, h, bpp int, p unsafe.Pointer) *Tensor {
	cType := C.halide_type_t{
		code:  C.halide_type_code_t(C.uint8_t(dataType.Code)), //C.uint8_t(dataType >> 24 & 0xFF),
		bits:  C.uint8_t(dataType.Bits),                       //C.uint8_t(dataType >> 16 & 0xFF),
		lanes: C.uint16_t(dataType.Lanes),                     //C.uint16_t(dataType & 0xFFFF),
	}

	cTensor := C.MNN_ImageProcess_createImageTensor(
		cType,
		C.int(w),
		C.int(h),
		C.int(bpp),
		p,
	)

	if cTensor == nil {
		return nil
	}

	return &Tensor{c: cTensor}
}

// SetPadding sets the padding value when wrap is ZERO
func (process *ImageProcess) SetPadding(value uint8) {
	if process != nil && process.c != nil {
		C.MNN_ImageProcess_setPadding(process.c, C.uint8_t(value))
	}
}

// SetDraw sets the ImageProcess to draw mode
func (process *ImageProcess) SetDraw() {
	if process != nil && process.c != nil {
		C.MNN_ImageProcess_setDraw(process.c)
	}
}

// Draw draws color to regions of the image
func (process *ImageProcess) Draw(img []byte, w, h, c int, regions []int, num int, color []uint8) {
	if process == nil || process.c == nil || img == nil || regions == nil || color == nil {
		return
	}

	cImg := (*C.uint8_t)(unsafe.Pointer(&img[0]))
	cRegions := (*C.int)(unsafe.Pointer(&regions[0]))
	cColor := (*C.uint8_t)(unsafe.Pointer(&color[0]))

	C.MNN_ImageProcess_draw(
		process.c,
		cImg,
		C.int(w),
		C.int(h),
		C.int(c),
		cRegions,
		C.int(num),
		cColor,
	)
}
