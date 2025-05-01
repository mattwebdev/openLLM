package training

import (
	"os"
	"testing"

	"openllm/pkg/model"
)

func TestTrainer(t *testing.T) {
	// Create temporary directory for checkpoints
	tempDir, err := os.MkdirTemp("", "trainer-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create test model
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
	testModel := model.NewTransformer(config)

	// Create trainer
	trainer, err := NewTrainer(testModel, &TrainerConfig{
		BatchSize:       32,
		LearningRate:    0.001,
		CheckpointDir:   tempDir,
		AccumulateSteps: 1,
	})
	if err != nil {
		t.Fatalf("Failed to create trainer: %v", err)
	}

	// Create test data
	data := make([]Batch, 100)
	for i := range data {
		data[i] = Batch{
			Input:  make([]int, 32),
			Target: make([]int, 32),
		}
	}

	// Test initial training
	if err := trainer.Train(data, 1); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Verify checkpoint was saved
	checkpoints, err := trainer.checkpointMgr.ListCheckpoints()
	if err != nil {
		t.Fatalf("Failed to list checkpoints: %v", err)
	}
	if len(checkpoints) == 0 {
		t.Error("No checkpoints were saved during training")
	}

	// Create new trainer with same checkpoint directory
	newTrainer, err := NewTrainer(testModel, &TrainerConfig{
		BatchSize:       32,
		LearningRate:    0.001,
		CheckpointDir:   tempDir,
		AccumulateSteps: 1,
	})
	if err != nil {
		t.Fatalf("Failed to create new trainer: %v", err)
	}

	// Train again - should resume from checkpoint
	if err := newTrainer.Train(data, 2); err != nil {
		t.Fatalf("Resume training failed: %v", err)
	}

	// Verify global step and epoch were restored
	if newTrainer.globalStep <= 0 {
		t.Error("Global step was not restored from checkpoint")
	}
	if newTrainer.currentEpoch <= 0 {
		t.Error("Current epoch was not restored from checkpoint")
	}
}

func TestTrainerErrors(t *testing.T) {
	// Test with invalid checkpoint directory
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
	testModel := model.NewTransformer(config)

	_, err := NewTrainer(testModel, &TrainerConfig{
		BatchSize:       32,
		LearningRate:    0.001,
		CheckpointDir:   "/invalid/dir",
		AccumulateSteps: 1,
	})
	if err == nil {
		t.Error("Expected error for invalid checkpoint directory")
	}

	// Test with nil model
	_, err = NewTrainer(nil, &TrainerConfig{
		BatchSize:       32,
		LearningRate:    0.001,
		CheckpointDir:   "checkpoints",
		AccumulateSteps: 1,
	})
	if err == nil {
		t.Error("Expected error for nil model")
	}

	// Test with invalid batch size
	_, err = NewTrainer(testModel, &TrainerConfig{
		BatchSize:       0,
		LearningRate:    0.001,
		CheckpointDir:   "checkpoints",
		AccumulateSteps: 1,
	})
	if err == nil {
		t.Error("Expected error for invalid batch size")
	}

	// Test with invalid learning rate
	_, err = NewTrainer(testModel, &TrainerConfig{
		BatchSize:       32,
		LearningRate:    -1,
		CheckpointDir:   "checkpoints",
		AccumulateSteps: 1,
	})
	if err == nil {
		t.Error("Expected error for invalid learning rate")
	}
}
