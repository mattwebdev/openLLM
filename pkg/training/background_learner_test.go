package training

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"openllm/pkg/model"
)

func TestBackgroundLearner(t *testing.T) {
	// Create temporary directories
	tempDataDir, err := os.MkdirTemp("", "background-learner-data-*")
	if err != nil {
		t.Fatalf("Failed to create temp data directory: %v", err)
	}
	defer os.RemoveAll(tempDataDir)

	tempCheckpointDir, err := os.MkdirTemp("", "background-learner-checkpoint-*")
	if err != nil {
		t.Fatalf("Failed to create temp checkpoint directory: %v", err)
	}
	defer os.RemoveAll(tempCheckpointDir)

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

	// Create test files
	testFiles := []struct {
		name    string
		content string
	}{
		{
			name:    "test1.md",
			content: "This is a test markdown file with some content for training.",
		},
		{
			name:    "test2.md",
			content: "Another test file with different content to process.",
		},
		{
			name:    "ignore.txt",
			content: "This file should be ignored.",
		},
	}

	for _, tf := range testFiles {
		path := filepath.Join(tempDataDir, tf.name)
		if err := os.WriteFile(path, []byte(tf.content), 0644); err != nil {
			t.Fatalf("Failed to create test file %s: %v", tf.name, err)
		}
	}

	// Create training channel
	trainingChan := make(chan string, 100)

	// Create background learner
	learner, err := NewBackgroundLearner(
		testModel,
		tempDataDir,
		tempCheckpointDir,
		32,
		0.001,
		trainingChan,
	)
	if err != nil {
		t.Fatalf("Failed to create background learner: %v", err)
	}

	// Test initial state
	if learner.IsActive() {
		t.Error("Background learner should not be active initially")
	}

	// Start background learning
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := learner.Start(ctx); err != nil {
		t.Fatalf("Failed to start background learner: %v", err)
	}

	// Verify learner is active
	if !learner.IsActive() {
		t.Error("Background learner should be active after starting")
	}

	// Wait for processing
	time.Sleep(6 * time.Second)

	// Check progress
	progress := learner.GetProgress()
	if progress["files_processed"].(int) != 2 {
		t.Errorf("Expected 2 files processed, got %d", progress["files_processed"].(int))
	}
	if progress["is_active"].(bool) != true {
		t.Error("Expected background learner to be active")
	}

	// Stop background learning
	learner.Stop()

	// Verify learner is stopped
	if learner.IsActive() {
		t.Error("Background learner should not be active after stopping")
	}
}

func TestBackgroundLearnerErrors(t *testing.T) {
	// Test with invalid data directory
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

	// Create training channel
	trainingChan := make(chan string, 100)

	_, err := NewBackgroundLearner(
		testModel,
		"/invalid/data/dir",
		"/invalid/checkpoint/dir",
		32,
		0.001,
		trainingChan,
	)
	if err == nil {
		t.Error("Expected error for invalid directories")
	}

	// Test with nil model
	_, err = NewBackgroundLearner(
		nil,
		"data",
		"checkpoints",
		32,
		0.001,
		trainingChan,
	)
	if err == nil {
		t.Error("Expected error for nil model")
	}

	// Test with invalid batch size
	_, err = NewBackgroundLearner(
		testModel,
		"data",
		"checkpoints",
		0,
		0.001,
		trainingChan,
	)
	if err == nil {
		t.Error("Expected error for invalid batch size")
	}

	// Test with invalid learning rate
	_, err = NewBackgroundLearner(
		testModel,
		"data",
		"checkpoints",
		32,
		-1,
		trainingChan,
	)
	if err == nil {
		t.Error("Expected error for invalid learning rate")
	}
}
