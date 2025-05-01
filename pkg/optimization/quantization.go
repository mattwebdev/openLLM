package optimization

import (
	"fmt"
	"math"

	"openllm/pkg/model"
)

// QuantizationConfig holds configuration for model quantization
type QuantizationConfig struct {
	Bits      int     // Number of bits for quantization
	Symmetric bool    // Whether to use symmetric quantization
	Calibrate bool    // Whether to calibrate scales dynamically
	ClipRatio float64 // Ratio for outlier clipping
}

// Quantizer handles model weight quantization
type Quantizer struct {
	config QuantizationConfig
	scales map[string]float32 // Quantization scales per layer
	zeros  map[string]int8    // Zero points per layer
}

// NewQuantizer creates a new quantizer instance
func NewQuantizer(config QuantizationConfig) *Quantizer {
	return &Quantizer{
		config: config,
		scales: make(map[string]float32),
		zeros:  make(map[string]int8),
	}
}

// QuantizeModel quantizes all weights in the model
func (q *Quantizer) QuantizeModel(model *model.Transformer) error {
	// Quantize embeddings
	q.quantizeMatrix("embeddings/word", model.WordEmbeddings.Data)
	q.quantizeMatrix("embeddings/position", model.PositionEmbeddings.Data)

	// Quantize each layer
	for i, layer := range model.Layers {
		prefix := fmt.Sprintf("layer_%d/", i)

		// Attention weights
		q.quantizeMatrix(prefix+"query", layer.QueryWeights.Data)
		q.quantizeMatrix(prefix+"key", layer.KeyWeights.Data)
		q.quantizeMatrix(prefix+"value", layer.ValueWeights.Data)
		q.quantizeMatrix(prefix+"output", layer.OutputWeights.Data)

		// FFN weights
		q.quantizeMatrix(prefix+"ffn1", layer.FFNWeights1.Data)
		q.quantizeMatrix(prefix+"ffn2", layer.FFNWeights2.Data)
	}

	return nil
}

// quantizeMatrix quantizes a weight matrix
func (q *Quantizer) quantizeMatrix(name string, data []float32) {
	if len(data) == 0 {
		return
	}

	// Find min/max values
	minVal, maxVal := float32(math.MaxFloat32), float32(-math.MaxFloat32)
	for _, v := range data {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	// Apply outlier clipping if enabled
	if q.config.ClipRatio > 0 {
		threshold := float32(q.config.ClipRatio * float64(maxVal-minVal))
		for i := range data {
			if data[i] > threshold {
				data[i] = threshold
			} else if data[i] < -threshold {
				data[i] = -threshold
			}
		}
	}

	// Calculate scale and zero point
	numLevels := 1 << uint(q.config.Bits)
	var scale float32
	var zeroPoint int8

	if q.config.Symmetric {
		// Symmetric quantization
		absMax := float32(math.Max(math.Abs(float64(minVal)), math.Abs(float64(maxVal))))
		scale = absMax / float32(numLevels/2-1)
		zeroPoint = 0
	} else {
		// Asymmetric quantization
		scale = (maxVal - minVal) / float32(numLevels-1)
		zeroPoint = int8(math.Round(float64(-minVal / scale)))
	}

	// Store quantization parameters
	q.scales[name] = scale
	q.zeros[name] = zeroPoint

	// Quantize weights
	for i := range data {
		quantized := int8(math.Round(float64(data[i]/scale))) + zeroPoint
		data[i] = float32(quantized-zeroPoint) * scale
	}
}

// DequantizeMatrix dequantizes a weight matrix
func (q *Quantizer) DequantizeMatrix(name string, quantized []int8) []float32 {
	scale := q.scales[name]
	zeroPoint := q.zeros[name]

	dequantized := make([]float32, len(quantized))
	for i, v := range quantized {
		dequantized[i] = float32(v-zeroPoint) * scale
	}
	return dequantized
}

// GetQuantizationStats returns statistics about the quantization
func (q *Quantizer) GetQuantizationStats() map[string]interface{} {
	stats := make(map[string]interface{})

	// Calculate average scale
	totalScale := float32(0)
	for _, scale := range q.scales {
		totalScale += scale
	}
	avgScale := totalScale / float32(len(q.scales))

	// Calculate scale distribution
	scaleVariance := float32(0)
	for _, scale := range q.scales {
		diff := scale - avgScale
		scaleVariance += diff * diff
	}
	scaleStdDev := float32(math.Sqrt(float64(scaleVariance / float32(len(q.scales)))))

	stats["num_quantized_layers"] = len(q.scales)
	stats["average_scale"] = avgScale
	stats["scale_stddev"] = scaleStdDev
	stats["bits"] = q.config.Bits
	stats["symmetric"] = q.config.Symmetric
	stats["calibrated"] = q.config.Calibrate

	return stats
}
