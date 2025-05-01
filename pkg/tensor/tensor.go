package tensor

import (
	"errors"
	"math"
)

// Tensor represents a multi-dimensional array
type Tensor struct {
	Shape []int
	Data  []float32
}

// NewTensor creates a new tensor with the given shape
func NewTensor(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return &Tensor{
		Shape: shape,
		Data:  make([]float32, size),
	}
}

// NewTensorFromData creates a new tensor with the given shape and data
func NewTensorFromData(shape []int, data []float32) (*Tensor, error) {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	if size != len(data) {
		return nil, errors.New("shape and data size mismatch")
	}
	return &Tensor{
		Shape: shape,
		Data:  data,
	}, nil
}

// Add performs element-wise addition
func (t *Tensor) Add(other *Tensor) (*Tensor, error) {
	if !t.ShapeEquals(other) {
		return nil, errors.New("shape mismatch")
	}
	result := NewTensor(t.Shape)
	for i := range t.Data {
		result.Data[i] = t.Data[i] + other.Data[i]
	}
	return result, nil
}

// Mul performs element-wise multiplication
func (t *Tensor) Mul(other *Tensor) (*Tensor, error) {
	if !t.ShapeEquals(other) {
		return nil, errors.New("shape mismatch")
	}
	result := NewTensor(t.Shape)
	for i := range t.Data {
		result.Data[i] = t.Data[i] * other.Data[i]
	}
	return result, nil
}

// MatMul performs matrix multiplication
func (t *Tensor) MatMul(other *Tensor) (*Tensor, error) {
	if len(t.Shape) != 2 || len(other.Shape) != 2 {
		return nil, errors.New("both tensors must be 2D")
	}
	if t.Shape[1] != other.Shape[0] {
		return nil, errors.New("inner dimensions must match")
	}

	result := NewTensor([]int{t.Shape[0], other.Shape[1]})
	for i := 0; i < t.Shape[0]; i++ {
		for j := 0; j < other.Shape[1]; j++ {
			sum := float32(0)
			for k := 0; k < t.Shape[1]; k++ {
				sum += t.Data[i*t.Shape[1]+k] * other.Data[k*other.Shape[1]+j]
			}
			result.Data[i*other.Shape[1]+j] = sum
		}
	}
	return result, nil
}

// ShapeEquals checks if two tensors have the same shape
func (t *Tensor) ShapeEquals(other *Tensor) bool {
	if len(t.Shape) != len(other.Shape) {
		return false
	}
	for i := range t.Shape {
		if t.Shape[i] != other.Shape[i] {
			return false
		}
	}
	return true
}

// Softmax applies softmax activation
func (t *Tensor) Softmax() *Tensor {
	result := NewTensor(t.Shape)
	max := float32(math.Inf(-1))
	for _, v := range t.Data {
		if v > max {
			max = v
		}
	}

	sum := float32(0)
	for i, v := range t.Data {
		result.Data[i] = float32(math.Exp(float64(v - max)))
		sum += result.Data[i]
	}

	for i := range result.Data {
		result.Data[i] /= sum
	}
	return result
}
