package training

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"

	"openllm/pkg/model"
)

// Trainer handles the training process
type Trainer struct {
	model         *model.Transformer
	batchSize     int
	learningRate  float32
	checkpointMgr *CheckpointManager
	currentEpoch  int
	globalStep    int
	metrics       map[string]float32
	scheduler     LRScheduler
	gradientAccum [][]float32
	accumSteps    int
	mu            sync.RWMutex
}

// TrainerConfig holds configuration for the trainer
type TrainerConfig struct {
	BatchSize       int
	LearningRate    float32
	CheckpointDir   string
	Scheduler       LRScheduler
	AccumulateSteps int
}

// NewTrainer creates a new trainer instance
func NewTrainer(model *model.Transformer, config *TrainerConfig) (*Trainer, error) {
	// Initialize checkpoint manager
	checkpointMgr, err := NewCheckpointManager(config.CheckpointDir, 5, 1000)
	if err != nil {
		return nil, fmt.Errorf("failed to create checkpoint manager: %v", err)
	}

	// Initialize gradient accumulation buffers
	accumSteps := config.AccumulateSteps
	if accumSteps < 1 {
		accumSteps = 1
	}

	return &Trainer{
		model:         model,
		batchSize:     config.BatchSize,
		learningRate:  config.LearningRate,
		checkpointMgr: checkpointMgr,
		metrics:       make(map[string]float32),
		scheduler:     config.Scheduler,
		accumSteps:    accumSteps,
	}, nil
}

// Batch represents a training batch
type Batch struct {
	Input  []int
	Target []int
}

// crossEntropyLoss calculates the cross-entropy loss between predictions and targets
func crossEntropyLoss(predictions, targets []float32) float32 {
	loss := float32(0.0)
	for i, target := range targets {
		// Add small epsilon to avoid log(0)
		pred := float32(math.Max(float64(predictions[i]), 1e-10))
		loss -= float32(math.Log(float64(pred))) * target
	}
	return loss / float32(len(targets))
}

// TrainStep performs a single training step
func (t *Trainer) TrainStep(batch *Batch) (float32, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Forward pass
	output, err := t.model.Forward(batch.Input)
	if err != nil {
		return 0, err
	}

	// Convert target indices to one-hot vectors
	targetOneHot := make([]float32, len(output))
	for _, idx := range batch.Target {
		if idx >= 0 && idx < len(targetOneHot) {
			targetOneHot[idx] = 1.0
		}
	}

	// Calculate loss
	loss := crossEntropyLoss(output, targetOneHot)

	// Calculate gradients
	gradOutput := make([]float32, len(output))
	for i := range output {
		gradOutput[i] = (output[i] - targetOneHot[i]) / float32(t.accumSteps)
	}

	// Initialize or accumulate gradients
	if t.gradientAccum == nil {
		t.gradientAccum = make([][]float32, len(t.model.Layers))
		for i := range t.model.Layers {
			t.gradientAccum[i] = make([]float32, len(gradOutput))
		}
	}

	// Accumulate gradients
	for i := range t.model.Layers {
		for j := range gradOutput {
			t.gradientAccum[i][j] += gradOutput[j]
		}
	}

	// Only update weights after accumulating enough gradients
	if (t.globalStep+1)%t.accumSteps == 0 {
		// Get current learning rate from scheduler
		currentLR := t.learningRate
		if t.scheduler != nil {
			currentLR = t.scheduler.GetLR(t.currentEpoch, t.globalStep)
		}

		// Apply accumulated gradients
		for i := range t.model.Layers {
			layer := t.model.Layers[i]
			grads := t.gradientAccum[i]

			// Update weights with accumulated gradients
			for j := range layer.QueryWeights.Data {
				layer.QueryWeights.Data[j] -= currentLR * grads[j]
				layer.KeyWeights.Data[j] -= currentLR * grads[j]
				layer.ValueWeights.Data[j] -= currentLR * grads[j]
				layer.OutputWeights.Data[j] -= currentLR * grads[j]
				layer.FFNWeights1.Data[j] -= currentLR * grads[j]
				layer.FFNWeights2.Data[j] -= currentLR * grads[j]
			}
		}

		// Reset accumulation buffers
		t.gradientAccum = nil
	}

	return loss, nil
}

// shuffleBatches randomly shuffles the training batches
func shuffleBatches(batches []Batch) {
	for i := len(batches) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		batches[i], batches[j] = batches[j], batches[i]
	}
}

// Train performs training on the given data for the specified number of epochs
func (t *Trainer) Train(data []Batch, epochs int) error {
	for epoch := 0; epoch < epochs; epoch++ {
		t.mu.Lock()
		t.currentEpoch = epoch
		t.mu.Unlock()

		// Shuffle the data
		shuffleBatches(data)

		// Process each batch
		for i, batch := range data {
			t.mu.Lock()
			t.globalStep = i
			t.mu.Unlock()

			// Perform training step
			loss, err := t.TrainStep(&batch)
			if err != nil {
				return fmt.Errorf("failed to perform training step: %v", err)
			}

			// Update metrics
			t.mu.Lock()
			t.metrics["loss"] = loss
			t.metrics["learning_rate"] = t.scheduler.GetLR(epoch, i)
			t.mu.Unlock()

			// Save checkpoint periodically
			if i%1000 == 0 {
				t.mu.RLock()
				if err := t.checkpointMgr.SaveCheckpoint(t.model, epoch, i, loss, t.metrics["learning_rate"], t.metrics); err != nil {
					log.Printf("Failed to save checkpoint: %v", err)
				}
				t.mu.RUnlock()
			}
		}
	}

	return nil
}
