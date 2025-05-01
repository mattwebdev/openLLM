package training

import (
	"os"
	"path/filepath"
	"testing"

	"openllm/pkg/model"
)

func TestCheckpointManager(t *testing.T) {
	// Create temporary directory for checkpoints
	tempDir, err := os.MkdirTemp("", "checkpoint-test-*")
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

	// Create checkpoint manager
	cm, err := NewCheckpointManager(tempDir, 3, 100)
	if err != nil {
		t.Fatalf("Failed to create checkpoint manager: %v", err)
	}

	// Test ShouldSave
	tests := []struct {
		step     int
		expected bool
	}{
		{99, false},
		{100, true},
		{150, false},
		{200, true},
		{300, true},
	}

	for _, tt := range tests {
		if got := cm.ShouldSave(tt.step); got != tt.expected {
			t.Errorf("ShouldSave(%d) = %v, want %v", tt.step, got, tt.expected)
		}
	}

	// Test SaveCheckpoint
	metrics := map[string]float32{
		"accuracy": 0.85,
		"loss":     0.15,
	}

	for i := 0; i < 5; i++ {
		err = cm.SaveCheckpoint(testModel, i, i*100, 0.5, 0.001, metrics)
		if err != nil {
			t.Errorf("SaveCheckpoint failed: %v", err)
		}
	}

	// Test ListCheckpoints
	checkpoints, err := cm.ListCheckpoints()
	if err != nil {
		t.Errorf("ListCheckpoints failed: %v", err)
	}

	// Should only keep 3 checkpoints (maxToKeep)
	if len(checkpoints) != 3 {
		t.Errorf("Expected 3 checkpoints, got %d", len(checkpoints))
	}

	// Test LoadLatestCheckpoint
	latest, err := cm.LoadLatestCheckpoint()
	if err != nil {
		t.Errorf("LoadLatestCheckpoint failed: %v", err)
	}

	if latest.Metadata.Epoch != 4 {
		t.Errorf("Expected latest checkpoint epoch to be 4, got %d", latest.Metadata.Epoch)
	}

	// Test LoadCheckpoint
	checkpoint, err := cm.LoadCheckpoint(3, 300)
	if err != nil {
		t.Errorf("LoadCheckpoint failed: %v", err)
	}

	if checkpoint.Metadata.Epoch != 3 {
		t.Errorf("Expected checkpoint epoch to be 3, got %d", checkpoint.Metadata.Epoch)
	}
}

func TestCheckpointManagerErrors(t *testing.T) {
	// Test with invalid directory
	_, err := NewCheckpointManager("/invalid/dir", 3, 100)
	if err == nil {
		t.Error("Expected error for invalid directory")
	}

	// Create temporary directory for checkpoints
	tempDir, err := os.MkdirTemp("", "checkpoint-error-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	cm, err := NewCheckpointManager(tempDir, 3, 100)
	if err != nil {
		t.Fatalf("Failed to create checkpoint manager: %v", err)
	}

	// Test loading non-existent checkpoint
	_, err = cm.LoadCheckpoint(1, 100)
	if err == nil {
		t.Error("Expected error when loading non-existent checkpoint")
	}

	// Test loading latest checkpoint when none exist
	_, err = cm.LoadLatestCheckpoint()
	if err == nil {
		t.Error("Expected error when loading latest checkpoint with no checkpoints")
	}

	// Test with invalid checkpoint file
	invalidFile := filepath.Join(tempDir, "checkpoint-0-0.gob")
	if err := os.WriteFile(invalidFile, []byte("invalid data"), 0644); err != nil {
		t.Fatalf("Failed to create invalid checkpoint file: %v", err)
	}

	_, err = cm.LoadCheckpoint(0, 0)
	if err == nil {
		t.Error("Expected error when loading invalid checkpoint file")
	}
}
