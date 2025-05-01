package training

import (
	"context"
	"fmt"
	"testing"
	"time"

	"openllm/pkg/model"
)

func TestDistributedTrainer(t *testing.T) {
	// Create a test model
	config := &model.Config{
		VocabSize:    1000,
		MaxSeqLength: 512,
		HiddenSize:   768,
		NumLayers:    6,
		NumHeads:     12,
		FFNDim:       3072,
		Dropout:      0.1,
		LayerNormEps: 1e-5,
	}
	model := model.NewTransformer(config)

	// Create a base trainer
	baseTrainer := &Trainer{
		model:        model,
		batchSize:    32,
		learningRate: 0.001,
	}

	// Test cases
	tests := []struct {
		name          string
		worldSize     int
		rank          int
		deviceID      int
		precisionMode PrecisionMode
	}{
		{
			name:          "Single GPU FP32",
			worldSize:     1,
			rank:          0,
			deviceID:      0,
			precisionMode: FP32,
		},
		{
			name:          "Single GPU FP16",
			worldSize:     1,
			rank:          0,
			deviceID:      0,
			precisionMode: FP16,
		},
		{
			name:          "Multi GPU FP32",
			worldSize:     2,
			rank:          0,
			deviceID:      0,
			precisionMode: FP32,
		},
		{
			name:          "Multi GPU FP16",
			worldSize:     2,
			rank:          0,
			deviceID:      0,
			precisionMode: FP16,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create test addresses
			addresses := make([]string, tt.worldSize)
			for i := 0; i < tt.worldSize; i++ {
				addresses[i] = fmt.Sprintf("127.0.0.1:%d", 8000+i)
			}

			// Create distributed trainer
			trainer, err := NewDistributedTrainer(
				baseTrainer,
				tt.worldSize,
				tt.rank,
				tt.deviceID,
				tt.precisionMode,
				addresses,
			)
			if err != nil {
				t.Fatalf("Failed to create distributed trainer: %v", err)
			}
			defer trainer.Close()

			// Create test data
			data := make([]Batch, 100)
			for i := range data {
				data[i] = Batch{
					Input:  make([]int, 32),
					Target: make([]int, 32),
				}
			}

			// Create context with timeout
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			// Start training
			err = trainer.StartDistributedTraining(ctx, data, 1)
			if err != nil {
				t.Fatalf("Training failed: %v", err)
			}

			// Verify model state
			if trainer.model == nil {
				t.Error("Model is nil after training")
			}

			// Verify gradient scaler
			if tt.precisionMode == FP16 || tt.precisionMode == BF16 {
				if trainer.scaler == nil {
					t.Error("Gradient scaler is nil for mixed precision training")
				}
				if trainer.scaler.scale == 0 {
					t.Error("Gradient scaler scale is 0")
				}
			}
		})
	}
}

func TestGradScaler(t *testing.T) {
	scaler := NewGradScaler()

	// Test initial values
	if scaler.scale != 1.0 {
		t.Errorf("Expected initial scale 1.0, got %f", scaler.scale)
	}

	// Test growth
	for i := 0; i < scaler.growthInterval; i++ {
		scaler.iterations++
	}
	if scaler.scale != scaler.growthFactor {
		t.Errorf("Expected scale %f after growth, got %f", scaler.growthFactor, scaler.scale)
	}

	// Test backoff
	for i := 0; i < scaler.backoffInterval; i++ {
		scaler.iterations++
	}
	expectedScale := scaler.growthFactor * scaler.backoffFactor
	if scaler.scale != expectedScale {
		t.Errorf("Expected scale %f after backoff, got %f", expectedScale, scaler.scale)
	}
}

func TestSplitData(t *testing.T) {
	data := make([]Batch, 100)
	for i := range data {
		data[i] = Batch{
			Input:  make([]int, 32),
			Target: make([]int, 32),
		}
	}

	tests := []struct {
		name      string
		worldSize int
		rank      int
		expected  int
	}{
		{
			name:      "Single rank",
			worldSize: 1,
			rank:      0,
			expected:  100,
		},
		{
			name:      "Two ranks, rank 0",
			worldSize: 2,
			rank:      0,
			expected:  50,
		},
		{
			name:      "Two ranks, rank 1",
			worldSize: 2,
			rank:      1,
			expected:  50,
		},
		{
			name:      "Three ranks, rank 0",
			worldSize: 3,
			rank:      0,
			expected:  34,
		},
		{
			name:      "Three ranks, rank 1",
			worldSize: 3,
			rank:      1,
			expected:  33,
		},
		{
			name:      "Three ranks, rank 2",
			worldSize: 3,
			rank:      2,
			expected:  33,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := splitData(data, tt.worldSize, tt.rank)
			if len(result) != tt.expected {
				t.Errorf("Expected %d batches, got %d", tt.expected, len(result))
			}
		})
	}
}
