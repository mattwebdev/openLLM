package training

import (
	"context"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"openllm/pkg/model"
)

// BackgroundLearner manages continuous learning when the system is idle
type BackgroundLearner struct {
	trainer        *Trainer
	dataDir        string
	mu             sync.RWMutex
	isActive       bool
	stopChan       chan struct{}
	maxBatchSize   int
	learningRate   float32
	checkpointDir  string
	processedFiles map[string]bool
	trainingChan   <-chan string
}

// NewBackgroundLearner creates a new background learning system
func NewBackgroundLearner(
	model *model.Transformer,
	dataDir string,
	checkpointDir string,
	maxBatchSize int,
	learningRate float32,
	trainingChan <-chan string,
) (*BackgroundLearner, error) {
	// Create trainer configuration with warmup and cosine annealing
	config := &TrainerConfig{
		BatchSize:       maxBatchSize,
		LearningRate:    learningRate,
		CheckpointDir:   checkpointDir,
		AccumulateSteps: 4, // Accumulate gradients over 4 steps
		Scheduler: NewCosineAnnealingLR(
			learningRate,
			learningRate*0.1, // Minimum LR is 10% of initial
			1000,             // Cycle every 1000 steps
		),
	}

	trainer, err := NewTrainer(model, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create trainer: %v", err)
	}

	return &BackgroundLearner{
		trainer:        trainer,
		dataDir:        dataDir,
		stopChan:       make(chan struct{}),
		maxBatchSize:   maxBatchSize,
		learningRate:   learningRate,
		checkpointDir:  checkpointDir,
		processedFiles: make(map[string]bool),
		trainingChan:   trainingChan,
	}, nil
}

// Start begins the background learning process
func (bl *BackgroundLearner) Start(ctx context.Context) error {
	bl.mu.Lock()
	if bl.isActive {
		bl.mu.Unlock()
		return fmt.Errorf("background learner is already active")
	}
	bl.isActive = true
	bl.mu.Unlock()

	go bl.learn(ctx)
	return nil
}

// Stop halts the background learning process
func (bl *BackgroundLearner) Stop() {
	bl.mu.Lock()
	defer bl.mu.Unlock()

	if bl.isActive {
		close(bl.stopChan)
		bl.isActive = false
	}
}

// IsActive returns whether background learning is currently active
func (bl *BackgroundLearner) IsActive() bool {
	bl.mu.RLock()
	defer bl.mu.RUnlock()
	return bl.isActive
}

// learn continuously processes papers and real-time content for training
func (bl *BackgroundLearner) learn(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Minute) // Check for new papers every 5 minutes
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-bl.stopChan:
			return
		case content := <-bl.trainingChan:
			// Process real-time content immediately
			if err := bl.processContent(content); err != nil {
				log.Printf("Error processing real-time content: %v", err)
			}
		case <-ticker.C:
			if err := bl.processPapers(ctx); err != nil {
				log.Printf("Error processing papers: %v", err)
			}
		}
	}
}

// processContent processes a single piece of content
func (bl *BackgroundLearner) processContent(content string) error {
	// Convert content to bytes for batch processing
	contentBytes := []byte(content)

	// Convert to batches
	batches, err := bl.convertToBatches(contentBytes)
	if err != nil {
		return fmt.Errorf("failed to convert content to batches: %v", err)
	}

	// Train on the batches for a single epoch
	if err := bl.trainer.Train(batches, 1); err != nil {
		return fmt.Errorf("failed to train on content: %v", err)
	}

	return nil
}

// processPapers reads and processes papers for training
func (bl *BackgroundLearner) processPapers(ctx context.Context) error {
	// Walk through all markdown and PDF files in the papers directory
	err := filepath.Walk(bl.dataDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip if already processed
		if bl.processedFiles[path] {
			return nil
		}

		// Only process markdown and PDF files
		ext := strings.ToLower(filepath.Ext(path))
		if ext != ".md" && ext != ".pdf" {
			return nil
		}

		// Read file content
		content, err := ioutil.ReadFile(path)
		if err != nil {
			return fmt.Errorf("failed to read file %s: %v", path, err)
		}

		// Convert content to training batches
		batches, err := bl.convertToBatches(content)
		if err != nil {
			return fmt.Errorf("failed to convert content to batches: %v", err)
		}

		// Train on the batches
		if err := bl.trainer.Train(batches, 1); err != nil {
			return fmt.Errorf("failed to train on file %s: %v", path, err)
		}

		// Mark file as processed
		bl.processedFiles[path] = true
		log.Printf("Successfully trained on %s", path)

		return nil
	})

	if err != nil {
		return fmt.Errorf("error walking papers directory: %v", err)
	}

	return nil
}

// convertToBatches converts raw content into training batches
func (bl *BackgroundLearner) convertToBatches(content []byte) ([]Batch, error) {
	// TODO: Implement proper tokenization and batching
	// For now, just create simple batches from the content
	text := string(content)
	words := strings.Fields(text)

	batches := make([]Batch, 0)
	for i := 0; i < len(words)-bl.maxBatchSize; i += bl.maxBatchSize {
		input := make([]int, bl.maxBatchSize)
		target := make([]int, bl.maxBatchSize)

		// Convert words to token IDs (simplified)
		for j := 0; j < bl.maxBatchSize; j++ {
			if i+j < len(words) {
				// TODO: Replace with proper tokenization
				input[j] = int(words[i+j][0]) % 1000 // Simple hash for demo
				if i+j+1 < len(words) {
					target[j] = int(words[i+j+1][0]) % 1000
				}
			}
		}

		batches = append(batches, Batch{
			Input:  input,
			Target: target,
		})
	}

	return batches, nil
}

// GetProgress returns training progress information
func (bl *BackgroundLearner) GetProgress() map[string]interface{} {
	bl.mu.RLock()
	defer bl.mu.RUnlock()

	checkpoints, _ := bl.trainer.checkpointMgr.ListCheckpoints()
	return map[string]interface{}{
		"is_active":        bl.isActive,
		"files_processed":  len(bl.processedFiles),
		"current_epoch":    bl.trainer.currentEpoch,
		"global_step":      bl.trainer.globalStep,
		"learning_rate":    bl.trainer.learningRate,
		"last_loss":        bl.trainer.metrics["loss"],
		"checkpoint_count": len(checkpoints),
	}
}
