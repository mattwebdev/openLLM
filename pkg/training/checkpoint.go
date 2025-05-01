package training

import (
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"openllm/pkg/model"
)

// CheckpointMetadata stores metadata about a saved checkpoint
type CheckpointMetadata struct {
	Timestamp    time.Time
	Epoch        int
	Step         int
	Loss         float32
	LearningRate float32
	Metrics      map[string]float32
}

// Checkpoint represents a saved model state
type Checkpoint struct {
	Model    *model.Transformer
	Metadata CheckpointMetadata
}

// CheckpointManager handles saving and loading of model checkpoints
type CheckpointManager struct {
	checkpointDir string
	maxToKeep     int
	saveEvery     int // Save every N steps
}

// NewCheckpointManager creates a new checkpoint manager
func NewCheckpointManager(checkpointDir string, maxToKeep int, saveEvery int) (*CheckpointManager, error) {
	// Create checkpoint directory if it doesn't exist
	if err := os.MkdirAll(checkpointDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create checkpoint directory: %v", err)
	}

	return &CheckpointManager{
		checkpointDir: checkpointDir,
		maxToKeep:     maxToKeep,
		saveEvery:     saveEvery,
	}, nil
}

// ShouldSave checks if we should save a checkpoint at the current step
func (cm *CheckpointManager) ShouldSave(step int) bool {
	return step%cm.saveEvery == 0
}

// SaveCheckpoint saves the model state and metadata to disk
func (cm *CheckpointManager) SaveCheckpoint(model *model.Transformer, epoch, step int, loss, lr float32, metrics map[string]float32) error {
	checkpoint := &Checkpoint{
		Model: model,
		Metadata: CheckpointMetadata{
			Timestamp:    time.Now(),
			Epoch:        epoch,
			Step:         step,
			Loss:         loss,
			LearningRate: lr,
			Metrics:      metrics,
		},
	}

	// Create checkpoint filename with timestamp and step
	filename := fmt.Sprintf("checkpoint-%d-%d.gob", epoch, step)
	filepath := filepath.Join(cm.checkpointDir, filename)

	// Open file for writing
	file, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("failed to create checkpoint file: %v", err)
	}
	defer file.Close()

	// Encode checkpoint using gob
	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(checkpoint); err != nil {
		return fmt.Errorf("failed to encode checkpoint: %v", err)
	}

	// Clean up old checkpoints if needed
	if err := cm.cleanOldCheckpoints(); err != nil {
		return fmt.Errorf("failed to clean old checkpoints: %v", err)
	}

	return nil
}

// LoadLatestCheckpoint loads the most recent checkpoint
func (cm *CheckpointManager) LoadLatestCheckpoint() (*Checkpoint, error) {
	// Get list of checkpoint files
	files, err := filepath.Glob(filepath.Join(cm.checkpointDir, "checkpoint-*.gob"))
	if err != nil {
		return nil, fmt.Errorf("failed to list checkpoint files: %v", err)
	}

	if len(files) == 0 {
		return nil, fmt.Errorf("no checkpoints found in %s", cm.checkpointDir)
	}

	// Find the latest checkpoint (last file when sorted)
	latestFile := files[len(files)-1]

	// Open file for reading
	file, err := os.Open(latestFile)
	if err != nil {
		return nil, fmt.Errorf("failed to open checkpoint file: %v", err)
	}
	defer file.Close()

	// Decode checkpoint using gob
	var checkpoint Checkpoint
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&checkpoint); err != nil {
		return nil, fmt.Errorf("failed to decode checkpoint: %v", err)
	}

	return &checkpoint, nil
}

// LoadCheckpoint loads a specific checkpoint by epoch and step
func (cm *CheckpointManager) LoadCheckpoint(epoch, step int) (*Checkpoint, error) {
	filename := fmt.Sprintf("checkpoint-%d-%d.gob", epoch, step)
	filepath := filepath.Join(cm.checkpointDir, filename)

	// Open file for reading
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open checkpoint file: %v", err)
	}
	defer file.Close()

	// Decode checkpoint using gob
	var checkpoint Checkpoint
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&checkpoint); err != nil {
		return nil, fmt.Errorf("failed to decode checkpoint: %v", err)
	}

	return &checkpoint, nil
}

// cleanOldCheckpoints removes old checkpoints if we have more than maxToKeep
func (cm *CheckpointManager) cleanOldCheckpoints() error {
	// Get list of checkpoint files
	files, err := filepath.Glob(filepath.Join(cm.checkpointDir, "checkpoint-*.gob"))
	if err != nil {
		return fmt.Errorf("failed to list checkpoint files: %v", err)
	}

	// If we have more files than maxToKeep, remove the oldest ones
	if len(files) > cm.maxToKeep {
		// Sort files by name (which includes timestamp)
		filesToDelete := files[:len(files)-cm.maxToKeep]
		for _, file := range filesToDelete {
			if err := os.Remove(file); err != nil {
				return fmt.Errorf("failed to remove old checkpoint %s: %v", file, err)
			}
		}
	}

	return nil
}

// ListCheckpoints returns a list of all available checkpoints
func (cm *CheckpointManager) ListCheckpoints() ([]CheckpointMetadata, error) {
	// Get list of checkpoint files
	files, err := filepath.Glob(filepath.Join(cm.checkpointDir, "checkpoint-*.gob"))
	if err != nil {
		return nil, fmt.Errorf("failed to list checkpoint files: %v", err)
	}

	metadataList := make([]CheckpointMetadata, 0, len(files))
	for _, file := range files {
		// Open file for reading
		f, err := os.Open(file)
		if err != nil {
			continue
		}

		// Decode checkpoint using gob
		var checkpoint Checkpoint
		decoder := gob.NewDecoder(f)
		if err := decoder.Decode(&checkpoint); err != nil {
			f.Close()
			continue
		}
		f.Close()

		metadataList = append(metadataList, checkpoint.Metadata)
	}

	return metadataList, nil
}
