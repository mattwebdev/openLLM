package optimization

import (
	"math"
	"testing"

	"openllm/pkg/model"
)

func TestNewPruner(t *testing.T) {
	config := PruningConfig{
		SparsityTarget: 0.5,
		Method:         "magnitude",
		BlockSize:      8,
		LayerDropout:   0.1,
		GradualSteps:   10,
	}

	p := NewPruner(config)

	if p.config.SparsityTarget != 0.5 {
		t.Errorf("Expected sparsity target 0.5, got %v", p.config.SparsityTarget)
	}
	if p.config.Method != "magnitude" {
		t.Error("Expected magnitude pruning method")
	}
	if len(p.masks) != 0 {
		t.Error("Masks map should be empty initially")
	}
	if len(p.stats) != 0 {
		t.Error("Stats map should be empty initially")
	}
}

func TestMagnitudePruning(t *testing.T) {
	tests := []struct {
		name           string
		data           []float32
		sparsityTarget float64
		wantSparsity   float64
		tolerance      float64
	}{
		{
			name:           "50% sparsity",
			data:           []float32{-1.0, -0.5, 0.1, 0.5, 1.0},
			sparsityTarget: 0.5,
			wantSparsity:   0.5,
			tolerance:      0.1,
		},
		{
			name:           "80% sparsity",
			data:           []float32{-2.0, -1.0, 0.0, 1.0, 2.0},
			sparsityTarget: 0.8,
			wantSparsity:   0.8,
			tolerance:      0.1,
		},
		{
			name:           "No pruning",
			data:           []float32{1.0, 1.0, 1.0},
			sparsityTarget: 0.0,
			wantSparsity:   0.0,
			tolerance:      0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := PruningConfig{
				SparsityTarget: tt.sparsityTarget,
				Method:         "magnitude",
			}
			p := NewPruner(config)

			// Make a copy of original data
			original := make([]float32, len(tt.data))
			copy(original, tt.data)

			// Apply pruning
			p.pruneMatrix("test", tt.data, tt.sparsityTarget)

			// Check sparsity
			zeros := 0
			for _, v := range tt.data {
				if v == 0 {
					zeros++
				}
			}
			gotSparsity := float64(zeros) / float64(len(tt.data))

			if math.Abs(gotSparsity-tt.wantSparsity) > tt.tolerance {
				t.Errorf("Got sparsity %v, want %v (±%v)", gotSparsity, tt.wantSparsity, tt.tolerance)
			}

			// Check mask
			mask := p.masks["test"]
			if len(mask) != len(tt.data) {
				t.Errorf("Mask length %v doesn't match data length %v", len(mask), len(tt.data))
			}

			// Check that masked values are zero
			for i, v := range tt.data {
				if !mask[i] && v != 0 {
					t.Errorf("Masked value at index %d should be zero, got %v", i, v)
				}
			}
		})
	}
}

func TestStructuredPruning(t *testing.T) {
	tests := []struct {
		name           string
		data           []float32
		blockSize      int
		sparsityTarget float64
		wantSparsity   float64
		tolerance      float64
	}{
		{
			name:           "2x2 blocks, 50% sparsity",
			data:           []float32{1.0, 1.0, 0.1, 0.1, 2.0, 2.0, 0.2, 0.2},
			blockSize:      2,
			sparsityTarget: 0.5,
			wantSparsity:   0.5,
			tolerance:      0.1,
		},
		{
			name:           "4x4 blocks, 75% sparsity",
			data:           make([]float32, 16),
			blockSize:      4,
			sparsityTarget: 0.75,
			wantSparsity:   0.75,
			tolerance:      0.1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := PruningConfig{
				SparsityTarget: tt.sparsityTarget,
				Method:         "structured",
				BlockSize:      tt.blockSize,
			}
			p := NewPruner(config)

			// Initialize test data if empty
			if len(tt.data) > 0 {
				for i := range tt.data {
					tt.data[i] = float32(i + 1)
				}
			}

			// Apply pruning
			p.pruneMatrix("test", tt.data, tt.sparsityTarget)

			// Check sparsity
			zeros := 0
			for _, v := range tt.data {
				if v == 0 {
					zeros++
				}
			}
			gotSparsity := float64(zeros) / float64(len(tt.data))

			if math.Abs(gotSparsity-tt.wantSparsity) > tt.tolerance {
				t.Errorf("Got sparsity %v, want %v (±%v)", gotSparsity, tt.wantSparsity, tt.tolerance)
			}

			// Check that zeros appear in blocks
			for i := 0; i < len(tt.data); i += tt.blockSize {
				end := i + tt.blockSize
				if end > len(tt.data) {
					end = len(tt.data)
				}

				// Check if block is all zeros or all non-zeros
				isZeroBlock := tt.data[i] == 0
				for j := i + 1; j < end; j++ {
					if (tt.data[j] == 0) != isZeroBlock {
						t.Errorf("Inconsistent block at indices %d-%d", i, end-1)
					}
				}
			}
		})
	}
}

func TestPruneModel(t *testing.T) {
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

	pruningConfig := PruningConfig{
		SparsityTarget: 0.5,
		Method:         "magnitude",
		BlockSize:      8,
		LayerDropout:   0.0,
		GradualSteps:   5,
	}
	p := NewPruner(pruningConfig)

	// Test gradual pruning
	for step := 0; step < pruningConfig.GradualSteps; step++ {
		err := p.PruneModel(testModel, step)
		if err != nil {
			t.Fatalf("Failed to prune model at step %d: %v", step, err)
		}

		// Get pruning stats
		stats := p.GetPruningStats()
		currentSparsity := stats["average_sparsity"].(float64)

		// Check that sparsity increases gradually
		expectedSparsity := pruningConfig.SparsityTarget * float64(step+1) / float64(pruningConfig.GradualSteps)
		if currentSparsity < 0 || currentSparsity > expectedSparsity+0.1 {
			t.Errorf("Step %d: unexpected sparsity %v, want <= %v", step, currentSparsity, expectedSparsity+0.1)
		}
	}

	// Check final stats
	stats := p.GetPruningStats()
	requiredStats := []string{
		"average_sparsity",
		"method",
		"target_sparsity",
		"block_size",
		"layer_dropout",
		"per_layer_stats",
	}

	for _, stat := range requiredStats {
		if _, exists := stats[stat]; !exists {
			t.Errorf("Missing required stat: %s", stat)
		}
	}
}
