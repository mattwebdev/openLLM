package optimization

import (
	"math"
	"testing"

	"openllm/pkg/model"
)

func TestNewQuantizer(t *testing.T) {
	config := QuantizationConfig{
		Bits:      8,
		Symmetric: true,
		Calibrate: false,
		ClipRatio: 0.0,
	}

	q := NewQuantizer(config)

	if q.config.Bits != 8 {
		t.Errorf("Expected 8 bits, got %d", q.config.Bits)
	}
	if !q.config.Symmetric {
		t.Error("Expected symmetric quantization")
	}
	if len(q.scales) != 0 {
		t.Error("Scales map should be empty initially")
	}
	if len(q.zeros) != 0 {
		t.Error("Zeros map should be empty initially")
	}
}

func TestQuantizeMatrix(t *testing.T) {
	tests := []struct {
		name      string
		data      []float32
		bits      int
		symmetric bool
		clipRatio float64
	}{
		{
			name:      "8-bit symmetric",
			data:      []float32{-1.0, -0.5, 0.0, 0.5, 1.0},
			bits:      8,
			symmetric: true,
			clipRatio: 0.0,
		},
		{
			name:      "4-bit asymmetric",
			data:      []float32{-1.0, -0.5, 0.0, 0.5, 1.0},
			bits:      4,
			symmetric: false,
			clipRatio: 0.0,
		},
		{
			name:      "8-bit with clipping",
			data:      []float32{-2.0, -1.0, 0.0, 1.0, 2.0},
			bits:      8,
			symmetric: true,
			clipRatio: 0.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := QuantizationConfig{
				Bits:      tt.bits,
				Symmetric: tt.symmetric,
				Calibrate: false,
				ClipRatio: tt.clipRatio,
			}
			q := NewQuantizer(config)

			// Make a copy of original data
			original := make([]float32, len(tt.data))
			copy(original, tt.data)

			// Quantize
			q.quantizeMatrix("test", tt.data)

			// Check quantization parameters
			if _, exists := q.scales["test"]; !exists {
				t.Error("Scale not stored")
			}
			if _, exists := q.zeros["test"]; !exists {
				t.Error("Zero point not stored")
			}

			// Check value ranges
			maxDiff := float32(0)
			for i := range tt.data {
				diff := math.Abs(float64(tt.data[i] - original[i]))
				if float32(diff) > maxDiff {
					maxDiff = float32(diff)
				}
			}

			// Maximum quantization error should be reasonable
			var maxAllowedError float32
			switch tt.bits {
			case 8:
				maxAllowedError = 1.0 / 128.0 // For 8-bit quantization
			case 4:
				maxAllowedError = 1.0 / 8.0 // For 4-bit quantization
			default:
				maxAllowedError = 1.0 / float32(math.Pow(2.0, float64(tt.bits-1)))
			}

			if maxDiff > maxAllowedError {
				t.Errorf("Quantization error too large: %v > %v", maxDiff, maxAllowedError)
			}
		})
	}
}

func TestQuantizeModel(t *testing.T) {
	// Create a small test model
	config := &model.Config{
		VocabSize:    100,
		MaxSeqLength: 32,
		HiddenSize:   64,
		NumLayers:    2,
		NumHeads:     4,
		FFNDim:       128,
		Dropout:      0.1,
		LayerNormEps: 1e-5,
	}
	testModel := model.NewTransformer(config)

	// Create quantizer
	qConfig := QuantizationConfig{
		Bits:      8,
		Symmetric: true,
		Calibrate: false,
		ClipRatio: 0.0,
	}
	q := NewQuantizer(qConfig)

	// Quantize model
	err := q.QuantizeModel(testModel)
	if err != nil {
		t.Fatalf("Failed to quantize model: %v", err)
	}

	// Check that all layers were quantized
	expectedLayers := 2 + // Embeddings (word + position)
		6*config.NumLayers // 6 weight matrices per transformer layer

	if len(q.scales) != expectedLayers {
		t.Errorf("Expected %d quantized layers, got %d", expectedLayers, len(q.scales))
	}

	// Get quantization stats
	stats := q.GetQuantizationStats()

	// Check stats
	requiredStats := []string{
		"num_quantized_layers",
		"average_scale",
		"scale_stddev",
		"bits",
		"symmetric",
		"calibrated",
	}

	for _, stat := range requiredStats {
		if _, exists := stats[stat]; !exists {
			t.Errorf("Missing required stat: %s", stat)
		}
	}
}

func TestDequantization(t *testing.T) {
	// Test data
	original := []float32{-1.0, -0.5, 0.0, 0.5, 1.0}

	// Create quantizer
	config := QuantizationConfig{
		Bits:      8,
		Symmetric: true,
		Calibrate: false,
		ClipRatio: 0.0,
	}
	q := NewQuantizer(config)

	// Quantize
	data := make([]float32, len(original))
	copy(data, original)
	q.quantizeMatrix("test", data)

	// Convert to int8
	quantized := make([]int8, len(data))
	scale := q.scales["test"]
	zero := q.zeros["test"]
	for i, v := range data {
		quantized[i] = int8(math.Round(float64(v/scale))) + zero
	}

	// Dequantize
	dequantized := q.DequantizeMatrix("test", quantized)

	// Compare with original
	maxDiff := float32(0)
	for i := range original {
		diff := math.Abs(float64(dequantized[i] - original[i]))
		if float32(diff) > maxDiff {
			maxDiff = float32(diff)
		}
	}

	// Check error bounds
	maxAllowedError := float32(1.0 / 128.0) // For 8-bit symmetric quantization
	if maxDiff > maxAllowedError {
		t.Errorf("Dequantization error too large: %v > %v", maxDiff, maxAllowedError)
	}
}
