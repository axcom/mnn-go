package mnn

/*
#include <Tensor_c.h>
#include <MNN/HalideRuntime.h>
typedef struct halide_type_t halide_type_t;
*/
import "C"
import "unsafe"

// DimensionType corresponds to MNN_DimensionType in C
const (
	DimensionType_TENSORFLOW = C.MNN_TENSORFLOW
	DimensionType_CAFFE      = C.MNN_CAFFE
	DimensionType_CAFFE_C4   = C.MNN_CAFFE_C4
)

// HandleDataType corresponds to MNN_HandleDataType in C
const (
	HandleDataType_NONE   = C.MNN_HANDLE_NONE
	HandleDataType_STRING = C.MNN_HANDLE_STRING
)

// MapType corresponds to MNN_MapType in C
const (
	MapType_WRITE = C.MNN_MAP_TENSOR_WRITE
	MapType_READ  = C.MNN_MAP_TENSOR_READ
)

// Tensor wraps MNN_Tensor in C
type Tensor struct {
	c *C.struct_MNN_Tensor
}

// CreateTensor creates a new tensor with the given dimension size and type
func CreateTensor(dimSize int, dimType int) *Tensor {
	cTensor := C.MNN_Tensor_Create(C.int(dimSize), C.enum_MNN_DimensionType(dimType))
	return &Tensor{c: cTensor}
}

// CreateTensorFromExisting creates a new tensor with the same shape as the given tensor
func CreateTensorFromExisting(tensor *Tensor, dimType int, allocMemory bool) *Tensor {
	cTensor := C.MNN_Tensor_CreateFromExisting(tensor.c, C.enum_MNN_DimensionType(dimType), B2C(allocMemory))
	return &Tensor{c: cTensor}
}

// CreateDeviceTensor creates a new device tensor with the given shape, type and dimension type
func CreateDeviceTensor(shape []int, dtype C.halide_type_t, dimType int) *Tensor {
	cShape := (*C.int)(unsafe.Pointer(&shape[0]))
	cShapeSize := C.int(len(shape))
	cTensor := C.MNN_Tensor_CreateDevice(cShape, cShapeSize, dtype, C.enum_MNN_DimensionType(dimType))
	return &Tensor{c: cTensor}
}

// CreateHostTensor creates a new host tensor with the given shape, type, data and dimension type
func CreateHostTensor(shape []int, dtype C.halide_type_t, data unsafe.Pointer, dimType int) *Tensor {
	cShape := (*C.int)(unsafe.Pointer(&shape[0]))
	cShapeSize := C.int(len(shape))
	cTensor := C.MNN_Tensor_CreateHost(cShape, cShapeSize, dtype, data, C.enum_MNN_DimensionType(dimType))
	return &Tensor{c: cTensor}
}

// CloneTensor creates a copy of the given tensor
func CloneTensor(src *Tensor, deepCopy bool) *Tensor {
	cTensor := C.MNN_Tensor_Clone(src.c, B2C(deepCopy))
	return &Tensor{c: cTensor}
}

// DestroyTensor destroys the given tensor
func DestroyTensor(tensor *Tensor) {
	C.MNN_Tensor_Destroy(tensor.c)
	tensor.c = nil
}

func (t *Tensor) Close(hostTensor *Tensor) {
	C.MNN_Tensor_Destroy(t.c)
	t.c = nil
}

// CopyFromHostTensor copies data from a host tensor to this tensor
func (t *Tensor) CopyFromHostTensor(hostTensor *Tensor) bool {
	result := C.MNN_Tensor_CopyFromHostTensor(t.c, hostTensor.c)
	return B2Go(result)
}

// CopyToHostTensor copies data from this tensor to a host tensor
func (t *Tensor) CopyToHostTensor(hostTensor *Tensor) bool {
	result := C.MNN_Tensor_CopyToHostTensor(t.c, hostTensor.c)
	return B2Go(result)
}

// CreateHostTensorFromDevice creates a host tensor from a device tensor
func CreateHostTensorFromDevice(deviceTensor *Tensor, copyData bool) *Tensor {
	cTensor := C.MNN_Tensor_CreateHostTensorFromDevice(deviceTensor.c, B2C(copyData))
	return &Tensor{c: cTensor}
}

// Buffer returns the halide_buffer_t of this tensor
func (t *Tensor) Buffer() *C.halide_buffer_t {
	return C.MNN_Tensor_Buffer(t.c)
}

// MutableBuffer returns a mutable halide_buffer_t of this tensor
func (t *Tensor) MutableBuffer() *C.halide_buffer_t {
	return C.MNN_Tensor_MutableBuffer(t.c)
}

// GetDimensionType returns the dimension type of this tensor
func (t *Tensor) GetDimensionType() int {
	return int(C.MNN_Tensor_GetDimensionType(t.c))
}

// GetHandleDataType returns the handle data type of this tensor
func (t *Tensor) GetHandleDataType() int {
	return int(C.MNN_Tensor_GetHandleDataType(t.c))
}

// SetType sets the data type of this tensor
func (t *Tensor) SetType(dataType int) {
	C.MNN_Tensor_SetType(t.c, C.int(dataType))
}

// GetType returns the data type of this tensor
func (t *Tensor) GetType() C.halide_type_t {
	//return C.MNN_Tensor_GetType(t.c)
	var halideType C.halide_type_t
	C.MNN_Tensor_GetHalideType(t.c, &halideType)
	return halideType
}

// Host returns the host data pointer of this tensor
func (t *Tensor) Host() unsafe.Pointer {
	return C.MNN_Tensor_Host(t.c)
}

// DeviceId returns the device ID of this tensor
func (t *Tensor) DeviceId() uint64 {
	return uint64(C.MNN_Tensor_DeviceId(t.c))
}

// Dimensions returns the number of dimensions of this tensor
func (t *Tensor) Dimensions() int {
	return int(C.MNN_Tensor_Dimensions(t.c))
}

// Shape returns the shape of this tensor
func (t *Tensor) Shape() []int {
	var cShapeSize C.int
	cShape := C.MNN_Tensor_Shape(t.c, &cShapeSize)
	defer C.MNN_Tensor_FreeShape(cShape)

	shapeSize := int(cShapeSize)
	shape := make([]int, shapeSize)

	if shapeSize > 0 {
		shapePtr := (*[1 << 30]C.int)(unsafe.Pointer(cShape))[:shapeSize:shapeSize]
		for i, v := range shapePtr {
			shape[i] = int(v)
		}
	}

	return shape
}

// Size returns the size of this tensor in bytes
func (t *Tensor) Size() int {
	return int(C.MNN_Tensor_Size(t.c))
}

// USize returns the size of this tensor in bytes as uint64
func (t *Tensor) USize() uint64 {
	return uint64(C.MNN_Tensor_USize(t.c))
}

// ElementSize returns the number of elements in this tensor
func (t *Tensor) ElementSize() int {
	return int(C.MNN_Tensor_ElementSize(t.c))
}

// Width returns the width of this tensor
func (t *Tensor) Width() int {
	return int(C.MNN_Tensor_Width(t.c))
}

// Height returns the height of this tensor
func (t *Tensor) Height() int {
	return int(C.MNN_Tensor_Height(t.c))
}

// Channel returns the channel count of this tensor
func (t *Tensor) Channel() int {
	return int(C.MNN_Tensor_Channel(t.c))
}

// Batch returns the batch count of this tensor
func (t *Tensor) Batch() int {
	return int(C.MNN_Tensor_Batch(t.c))
}

// Stride returns the stride of the given dimension
func (t *Tensor) Stride(index int) int {
	return int(C.MNN_Tensor_Stride(t.c, C.int(index)))
}

// Length returns the length of the given dimension
func (t *Tensor) Length(index int) int {
	return int(C.MNN_Tensor_Length(t.c, C.int(index)))
}

// SetStride sets the stride of the given dimension
func (t *Tensor) SetStride(index int, stride int) {
	C.MNN_Tensor_SetStride(t.c, C.int(index), C.int(stride))
}

// SetLength sets the length of the given dimension
func (t *Tensor) SetLength(index int, length int) {
	C.MNN_Tensor_SetLength(t.c, C.int(index), C.int(length))
}

// GetDeviceInfo gets device information
func (t *Tensor) GetDeviceInfo(dst unsafe.Pointer, forwardType int) bool {
	result := C.MNN_Tensor_GetDeviceInfo(t.c, dst, C.int(forwardType))
	return B2Go(result)
}

// Print prints the tensor data
func (t *Tensor) Print() {
	C.MNN_Tensor_Print(t.c)
}

// PrintShape prints the tensor shape
func (t *Tensor) PrintShape() {
	C.MNN_Tensor_PrintShape(t.c)
}

// Map maps the tensor for reading or writing
func (t *Tensor) Map(mapType int, dimType int) unsafe.Pointer {
	return C.MNN_Tensor_Map(t.c, C.enum_MNN_MapType(mapType), C.enum_MNN_DimensionType(dimType))
}

// Unmap unmaps the tensor
func (t *Tensor) Unmap(mapType int, dimType int, mapPtr unsafe.Pointer) {
	C.MNN_Tensor_Unmap(t.c, C.enum_MNN_MapType(mapType), C.enum_MNN_DimensionType(dimType), mapPtr)
}

// Wait waits until the tensor is ready to read or write
func (t *Tensor) Wait(mapType int, finish bool) int {
	return int(C.MNN_Tensor_Wait(t.c, C.enum_MNN_MapType(mapType), B2C(finish)))
}

// SetDevicePtr sets the device pointer of the tensor
func (t *Tensor) SetDevicePtr(devicePtr unsafe.Pointer, memoryType int) bool {
	result := C.MNN_Tensor_SetDevicePtr(t.c, devicePtr, C.int(memoryType))
	return B2Go(result)
}
