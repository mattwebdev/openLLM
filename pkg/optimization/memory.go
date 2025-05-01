package optimization

import (
	"fmt"
	"math"
	"sync"

	"openllm/pkg/model"
)

// MemoryConfig holds configuration for memory optimization
type MemoryConfig struct {
	EnableGradientCheckpointing bool    // Enable gradient checkpointing
	CheckpointEvery             int     // Number of layers between checkpoints
	EnableMemoryEfficientAttn   bool    // Enable memory-efficient attention
	MaxActivationMemory         int64   // Maximum memory for activations in bytes
	MemoryBuffer                float64 // Memory buffer ratio (0.0 to 1.0)
}

// MemoryOptimizer handles memory optimization
type MemoryOptimizer struct {
	config MemoryConfig
	stats  map[string]interface{}
	mu     sync.RWMutex

	// Gradient checkpointing
	checkpoints    map[int][]float32
	checkpointLocs []int

	// Memory tracking
	currentMemory int64
	peakMemory    int64
}

// NewMemoryOptimizer creates a new memory optimizer
func NewMemoryOptimizer(config MemoryConfig) *MemoryOptimizer {
	return &MemoryOptimizer{
		config:      config,
		stats:       make(map[string]interface{}),
		checkpoints: make(map[int][]float32),
	}
}

// OptimizeMemory applies memory optimizations to the model
func (m *MemoryOptimizer) OptimizeMemory(model *model.Transformer) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.config.EnableGradientCheckpointing {
		if err := m.setupGradientCheckpointing(model); err != nil {
			return fmt.Errorf("failed to setup gradient checkpointing: %v", err)
		}
	}

	if m.config.EnableMemoryEfficientAttn {
		if err := m.optimizeAttention(model); err != nil {
			return fmt.Errorf("failed to optimize attention: %v", err)
		}
	}

	return nil
}

// setupGradientCheckpointing configures gradient checkpointing
func (m *MemoryOptimizer) setupGradientCheckpointing(model *model.Transformer) error {
	// Calculate checkpoint locations
	numLayers := len(model.Layers)
	if numLayers == 0 {
		return fmt.Errorf("model has no layers")
	}

	// Determine checkpoint frequency
	checkpointEvery := m.config.CheckpointEvery
	if checkpointEvery <= 0 {
		checkpointEvery = numLayers / 4 // Default to 4 checkpoints
		if checkpointEvery == 0 {
			checkpointEvery = 1
		}
	}

	// Set checkpoint locations
	m.checkpointLocs = make([]int, 0)
	for i := checkpointEvery - 1; i < numLayers; i += checkpointEvery {
		m.checkpointLocs = append(m.checkpointLocs, i)
	}

	m.stats["num_checkpoints"] = len(m.checkpointLocs)
	m.stats["checkpoint_layers"] = m.checkpointLocs
	return nil
}

// optimizeAttention implements memory-efficient attention
func (m *MemoryOptimizer) optimizeAttention(model *model.Transformer) error {
	for i, layer := range model.Layers {
		// Convert to memory-efficient attention
		if err := m.convertToFlashAttention(layer); err != nil {
			return fmt.Errorf("failed to convert layer %d to flash attention: %v", i, err)
		}
	}

	return nil
}

// convertToFlashAttention converts standard attention to flash attention
func (m *MemoryOptimizer) convertToFlashAttention(layer *model.TransformerLayer) error {
	// Calculate memory requirements
	qkvSize := len(layer.QueryWeights.Data) + len(layer.KeyWeights.Data) + len(layer.ValueWeights.Data)
	memoryNeeded := int64(qkvSize * 4) // 4 bytes per float32

	// Check if we have enough memory budget
	if m.currentMemory+memoryNeeded > m.config.MaxActivationMemory {
		return fmt.Errorf("insufficient memory for flash attention conversion")
	}

	// Update memory tracking
	m.currentMemory += memoryNeeded
	if m.currentMemory > m.peakMemory {
		m.peakMemory = m.currentMemory
	}

	// Optimize attention weights for tiled computation
	m.optimizeForTiledComputation(layer)

	return nil
}

// optimizeForTiledComputation reorganizes weights for tiled attention
func (m *MemoryOptimizer) optimizeForTiledComputation(layer *model.TransformerLayer) {
	// Calculate optimal tile size based on available memory
	tileSize := m.calculateOptimalTileSize(layer)

	// Reorganize weights for tiled computation
	m.reorganizeWeights(layer, tileSize)

	// Update stats
	m.stats["tile_size"] = tileSize
}

// calculateOptimalTileSize determines the best tile size
func (m *MemoryOptimizer) calculateOptimalTileSize(layer *model.TransformerLayer) int {
	// Get attention dimensions
	headDim := len(layer.QueryWeights.Data) / layer.NumHeads
	seqLen := layer.MaxSeqLength

	// Calculate memory per tile
	memoryPerElement := 4 // bytes per float32
	availableMemory := m.config.MaxActivationMemory - m.currentMemory

	// Calculate maximum tile size that fits in memory
	// Formula: tileSize^2 * headDim * numHeads * 3 (Q,K,V) * memoryPerElement <= availableMemory
	maxTileSize := int(math.Sqrt(float64(availableMemory) / float64(headDim*layer.NumHeads*3*memoryPerElement)))

	// Round down to nearest power of 2 for efficiency
	tileSize := 1
	for tileSize*2 <= maxTileSize && tileSize*2 <= seqLen {
		tileSize *= 2
	}

	return tileSize
}

// reorganizeWeights reorganizes attention weights for tiled computation
func (m *MemoryOptimizer) reorganizeWeights(layer *model.TransformerLayer, tileSize int) {
	// Reorganize query weights
	qWeights := make([]float32, len(layer.QueryWeights.Data))
	copy(qWeights, layer.QueryWeights.Data)
	m.tileWeights(qWeights, layer.NumHeads, tileSize)
	layer.QueryWeights.Data = qWeights

	// Reorganize key weights
	kWeights := make([]float32, len(layer.KeyWeights.Data))
	copy(kWeights, layer.KeyWeights.Data)
	m.tileWeights(kWeights, layer.NumHeads, tileSize)
	layer.KeyWeights.Data = kWeights

	// Reorganize value weights
	vWeights := make([]float32, len(layer.ValueWeights.Data))
	copy(vWeights, layer.ValueWeights.Data)
	m.tileWeights(vWeights, layer.NumHeads, tileSize)
	layer.ValueWeights.Data = vWeights
}

// tileWeights reorganizes weights into tiles
func (m *MemoryOptimizer) tileWeights(weights []float32, numHeads, tileSize int) {
	headDim := len(weights) / numHeads
	seqLen := len(weights) / headDim

	// Create temporary buffer
	temp := make([]float32, len(weights))
	copy(temp, weights)

	// Reorganize into tiles
	idx := 0
	for h := 0; h < numHeads; h++ {
		for i := 0; i < seqLen; i += tileSize {
			for j := 0; j < seqLen; j += tileSize {
				// Copy tile
				for ti := 0; ti < tileSize && i+ti < seqLen; ti++ {
					for tj := 0; tj < tileSize && j+tj < seqLen; tj++ {
						srcIdx := h*seqLen*headDim + (i+ti)*headDim + j + tj
						weights[idx] = temp[srcIdx]
						idx++
					}
				}
			}
		}
	}
}

// SaveCheckpoint saves a layer's state for gradient checkpointing
func (m *MemoryOptimizer) SaveCheckpoint(layerIdx int, state []float32) {
	m.mu.Lock()
	defer m.mu.Unlock()

	checkpoint := make([]float32, len(state))
	copy(checkpoint, state)
	m.checkpoints[layerIdx] = checkpoint
}

// LoadCheckpoint loads a layer's state from a checkpoint
func (m *MemoryOptimizer) LoadCheckpoint(layerIdx int) ([]float32, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	state, exists := m.checkpoints[layerIdx]
	if !exists {
		return nil, false
	}

	checkpoint := make([]float32, len(state))
	copy(checkpoint, state)
	return checkpoint, true
}

// GetMemoryStats returns statistics about memory usage
func (m *MemoryOptimizer) GetMemoryStats() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats := make(map[string]interface{})
	for k, v := range m.stats {
		stats[k] = v
	}

	stats["current_memory"] = m.currentMemory
	stats["peak_memory"] = m.peakMemory
	stats["memory_limit"] = m.config.MaxActivationMemory
	stats["memory_utilization"] = float64(m.currentMemory) / float64(m.config.MaxActivationMemory)
	stats["gradient_checkpointing_enabled"] = m.config.EnableGradientCheckpointing
	stats["memory_efficient_attn_enabled"] = m.config.EnableMemoryEfficientAttn

	return stats
}

// ClearCheckpoints removes all saved checkpoints
func (m *MemoryOptimizer) ClearCheckpoints() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.checkpoints = make(map[int][]float32)
	m.checkpointLocs = nil
}
