package training

import (
	"context"
	"encoding/gob"
	"fmt"
	"net"
	"sync"

	"openllm/pkg/model"
)

// PrecisionMode defines the floating-point precision mode
type PrecisionMode int

const (
	// FP32 represents full 32-bit floating point precision
	FP32 PrecisionMode = iota
	// FP16 represents 16-bit floating point precision
	FP16
	// BF16 represents Brain Floating Point 16-bit precision
	BF16
)

// Message types for network communication
const (
	msgGradients = iota
	msgModelState
)

// Message represents a network message
type Message struct {
	Type    int
	Rank    int
	Payload interface{}
}

// DistributedTrainer handles distributed training across multiple GPUs
type DistributedTrainer struct {
	*Trainer
	worldSize     int
	rank          int
	deviceID      int
	gradients     chan []float32
	stopChan      chan struct{}
	modelStates   map[int]*model.Transformer
	mu            sync.Mutex
	listener      net.Listener
	connections   []net.Conn
	precisionMode PrecisionMode
	scaler        *GradScaler
}

// GradScaler handles gradient scaling for mixed precision training
type GradScaler struct {
	scale           float32
	growth          float32
	backoff         float32
	growthInterval  int
	backoffInterval int
	growthFactor    float32
	backoffFactor   float32
	iterations      int
}

// NewGradScaler creates a new gradient scaler
func NewGradScaler() *GradScaler {
	return &GradScaler{
		scale:           1.0,
		growth:          2.0,
		backoff:         0.5,
		growthInterval:  2000,
		backoffInterval: 2000,
		growthFactor:    2.0,
		backoffFactor:   0.5,
		iterations:      0,
	}
}

// NewDistributedTrainer creates a new distributed trainer instance
func NewDistributedTrainer(
	baseTrainer *Trainer,
	worldSize int,
	rank int,
	deviceID int,
	precisionMode PrecisionMode,
	addresses []string,
) (*DistributedTrainer, error) {
	trainer := &DistributedTrainer{
		Trainer:       baseTrainer,
		worldSize:     worldSize,
		rank:          rank,
		deviceID:      deviceID,
		gradients:     make(chan []float32, worldSize),
		stopChan:      make(chan struct{}),
		modelStates:   make(map[int]*model.Transformer),
		precisionMode: precisionMode,
		scaler:        NewGradScaler(),
		connections:   make([]net.Conn, worldSize),
	}

	// Start listening for connections
	listener, err := net.Listen("tcp", addresses[rank])
	if err != nil {
		return nil, fmt.Errorf("failed to start listener: %v", err)
	}
	trainer.listener = listener

	// Connect to other ranks
	for i, addr := range addresses {
		if i == rank {
			continue
		}
		conn, err := net.Dial("tcp", addr)
		if err != nil {
			return nil, fmt.Errorf("failed to connect to rank %d: %v", i, err)
		}
		trainer.connections[i] = conn
	}

	// Start message handler
	go trainer.handleMessages()

	return trainer, nil
}

// handleMessages handles incoming network messages
func (dt *DistributedTrainer) handleMessages() {
	for {
		select {
		case <-dt.stopChan:
			return
		default:
			conn, err := dt.listener.Accept()
			if err != nil {
				continue
			}
			go dt.handleConnection(conn)
		}
	}
}

// handleConnection handles a single connection
func (dt *DistributedTrainer) handleConnection(conn net.Conn) {
	defer conn.Close()
	dec := gob.NewDecoder(conn)

	var msg Message
	if err := dec.Decode(&msg); err != nil {
		return
	}

	switch msg.Type {
	case msgGradients:
		if grads, ok := msg.Payload.([]float32); ok {
			dt.gradients <- grads
		}
	case msgModelState:
		if modelState, ok := msg.Payload.(*model.Transformer); ok {
			dt.modelStates[msg.Rank] = modelState
		}
	}
}

// sendMessage sends a message to a specific rank
func (dt *DistributedTrainer) sendMessage(rank int, msgType int, payload interface{}) error {
	conn := dt.connections[rank]
	if conn == nil {
		return fmt.Errorf("no connection to rank %d", rank)
	}

	enc := gob.NewEncoder(conn)
	msg := Message{
		Type:    msgType,
		Rank:    dt.rank,
		Payload: payload,
	}
	return enc.Encode(msg)
}

// synchronizeModelStates synchronizes model states across all ranks
func (dt *DistributedTrainer) synchronizeModelStates() error {
	// Get gradients from current model
	gradients := dt.getGradients()

	// Send gradients to all ranks
	for i := 0; i < dt.worldSize; i++ {
		if i == dt.rank {
			continue
		}
		if err := dt.sendMessage(i, msgGradients, gradients); err != nil {
			return fmt.Errorf("failed to send gradients to rank %d: %v", i, err)
		}
	}

	// Collect gradients from all ranks
	allGradients := make([][]float32, dt.worldSize)
	allGradients[dt.rank] = gradients

	// Wait for gradients from other ranks
	for i := 0; i < dt.worldSize-1; i++ {
		grads := <-dt.gradients
		allGradients[i] = grads
	}

	// Average gradients
	avgGradients := make([]float32, len(gradients))
	for i := range avgGradients {
		for j := 0; j < dt.worldSize; j++ {
			avgGradients[i] += allGradients[j][i]
		}
		avgGradients[i] /= float32(dt.worldSize)
	}

	// Apply averaged gradients
	dt.applyGradients(avgGradients)

	// Broadcast updated model state
	for i := 0; i < dt.worldSize; i++ {
		if i == dt.rank {
			continue
		}
		if err := dt.sendMessage(i, msgModelState, dt.model); err != nil {
			return fmt.Errorf("failed to send model state to rank %d: %v", i, err)
		}
	}

	return nil
}

// Close cleans up network resources
func (dt *DistributedTrainer) Close() error {
	close(dt.stopChan)
	if dt.listener != nil {
		dt.listener.Close()
	}
	for _, conn := range dt.connections {
		if conn != nil {
			conn.Close()
		}
	}
	return nil
}

// getGradients returns the current gradients
func (dt *DistributedTrainer) getGradients() []float32 {
	// Collect gradients from all model parameters
	var gradients []float32

	// Collect gradients from embeddings
	gradients = append(gradients, dt.model.WordEmbeddings.Data...)
	gradients = append(gradients, dt.model.PositionEmbeddings.Data...)

	// Collect gradients from layers
	for _, layer := range dt.model.Layers {
		gradients = append(gradients, layer.QueryWeights.Data...)
		gradients = append(gradients, layer.KeyWeights.Data...)
		gradients = append(gradients, layer.ValueWeights.Data...)
		gradients = append(gradients, layer.OutputWeights.Data...)
		gradients = append(gradients, layer.FFNWeights1.Data...)
		gradients = append(gradients, layer.FFNWeights2.Data...)
		gradients = append(gradients, layer.LayerNorm1.Data...)
		gradients = append(gradients, layer.LayerNorm2.Data...)
	}

	// Collect gradients from output layer
	gradients = append(gradients, dt.model.OutputLayer.Data...)

	return gradients
}

// applyGradients applies the averaged gradients to the model with mixed precision support
func (dt *DistributedTrainer) applyGradients(gradients []float32) {
	// Scale gradients for mixed precision
	if dt.precisionMode == FP16 || dt.precisionMode == BF16 {
		for i := range gradients {
			gradients[i] /= dt.scaler.scale
		}
	}

	offset := 0

	// Apply gradients to embeddings
	wordEmbeddingsSize := len(dt.model.WordEmbeddings.Data)
	copy(dt.model.WordEmbeddings.Data, gradients[offset:offset+wordEmbeddingsSize])
	offset += wordEmbeddingsSize

	positionEmbeddingsSize := len(dt.model.PositionEmbeddings.Data)
	copy(dt.model.PositionEmbeddings.Data, gradients[offset:offset+positionEmbeddingsSize])
	offset += positionEmbeddingsSize

	// Apply gradients to layers
	for _, layer := range dt.model.Layers {
		// Apply gradients to self-attention weights
		querySize := len(layer.QueryWeights.Data)
		copy(layer.QueryWeights.Data, gradients[offset:offset+querySize])
		offset += querySize

		keySize := len(layer.KeyWeights.Data)
		copy(layer.KeyWeights.Data, gradients[offset:offset+keySize])
		offset += keySize

		valueSize := len(layer.ValueWeights.Data)
		copy(layer.ValueWeights.Data, gradients[offset:offset+valueSize])
		offset += valueSize

		outputSize := len(layer.OutputWeights.Data)
		copy(layer.OutputWeights.Data, gradients[offset:offset+outputSize])
		offset += outputSize

		// Apply gradients to feed-forward network weights
		ffn1Size := len(layer.FFNWeights1.Data)
		copy(layer.FFNWeights1.Data, gradients[offset:offset+ffn1Size])
		offset += ffn1Size

		ffn2Size := len(layer.FFNWeights2.Data)
		copy(layer.FFNWeights2.Data, gradients[offset:offset+ffn2Size])
		offset += ffn2Size

		// Apply gradients to layer normalization parameters
		norm1Size := len(layer.LayerNorm1.Data)
		copy(layer.LayerNorm1.Data, gradients[offset:offset+norm1Size])
		offset += norm1Size

		norm2Size := len(layer.LayerNorm2.Data)
		copy(layer.LayerNorm2.Data, gradients[offset:offset+norm2Size])
		offset += norm2Size
	}

	// Apply gradients to output layer
	outputSize := len(dt.model.OutputLayer.Data)
	copy(dt.model.OutputLayer.Data, gradients[offset:offset+outputSize])

	// Update scaler
	dt.scaler.iterations++
	if dt.scaler.iterations%dt.scaler.growthInterval == 0 {
		dt.scaler.scale *= dt.scaler.growthFactor
	}
	if dt.scaler.iterations%dt.scaler.backoffInterval == 0 {
		dt.scaler.scale *= dt.scaler.backoffFactor
	}
}

// StartDistributedTraining starts the distributed training process
func (dt *DistributedTrainer) StartDistributedTraining(ctx context.Context, data []Batch, epochs int) error {
	// Initialize model states for each rank
	for i := 0; i < dt.worldSize; i++ {
		dt.modelStates[i] = dt.model.Clone()
	}

	// Start training loop
	for epoch := 0; epoch < epochs; epoch++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			if err := dt.trainEpoch(data); err != nil {
				return fmt.Errorf("error in epoch %d: %v", epoch, err)
			}
		}
	}

	// Stop training
	close(dt.stopChan)
	return nil
}

// trainEpoch performs one epoch of distributed training
func (dt *DistributedTrainer) trainEpoch(data []Batch) error {
	// Split data among ranks
	rankData := splitData(data, dt.worldSize, dt.rank)

	// Process local data
	for i := 0; i < len(rankData); i += dt.batchSize {
		end := i + dt.batchSize
		if end > len(rankData) {
			end = len(rankData)
		}

		batch := rankData[i:end]
		loss, err := dt.TrainStep(&batch[0])
		if err != nil {
			return err
		}

		// Log loss for monitoring
		if dt.rank == 0 {
			fmt.Printf("Rank %d: Loss = %.4f\n", dt.rank, loss)
		}

		// Collect gradients from all ranks
		dt.gradients <- dt.getGradients()
	}

	// Synchronize model states
	return dt.synchronizeModelStates()
}

// splitData splits the data among ranks
func splitData(data []Batch, worldSize, rank int) []Batch {
	chunkSize := len(data) / worldSize
	start := rank * chunkSize
	end := start + chunkSize
	if rank == worldSize-1 {
		end = len(data)
	}
	return data[start:end]
}
