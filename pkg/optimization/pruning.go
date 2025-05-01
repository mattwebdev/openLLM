package optimization

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"strings"

	"openllm/pkg/model"
)

// PruningConfig holds configuration for model pruning
type PruningConfig struct {
	SparsityTarget float64 // Target sparsity ratio (0.0 to 1.0)
	Method         string  // "magnitude" or "structured"
	BlockSize      int     // Block size for structured pruning
	LayerDropout   float64 // Probability of dropping entire layers
	GradualSteps   int     // Number of steps for gradual pruning
}

// Pruner handles model weight pruning
type Pruner struct {
	config PruningConfig
	masks  map[string][]bool  // Pruning masks per layer
	stats  map[string]float64 // Pruning statistics
}

// NewPruner creates a new pruner instance
func NewPruner(config PruningConfig) *Pruner {
	return &Pruner{
		config: config,
		masks:  make(map[string][]bool),
		stats:  make(map[string]float64),
	}
}

// PruneModel applies pruning to all model weights
func (p *Pruner) PruneModel(model *model.Transformer, step int) error {
	// Calculate current sparsity target based on gradual steps
	currentTarget := p.calculateCurrentSparsity(step)

	// Prune embeddings
	p.pruneMatrix("embeddings/word", model.WordEmbeddings.Data, currentTarget)
	p.pruneMatrix("embeddings/position", model.PositionEmbeddings.Data, currentTarget)

	// Prune transformer layers
	for i, layer := range model.Layers {
		// Check for layer dropout
		if p.shouldDropLayer() {
			p.dropLayer(layer)
			continue
		}

		prefix := fmt.Sprintf("layer_%d/", i)

		// Prune attention weights
		p.pruneMatrix(prefix+"query", layer.QueryWeights.Data, currentTarget)
		p.pruneMatrix(prefix+"key", layer.KeyWeights.Data, currentTarget)
		p.pruneMatrix(prefix+"value", layer.ValueWeights.Data, currentTarget)
		p.pruneMatrix(prefix+"output", layer.OutputWeights.Data, currentTarget)

		// Prune FFN weights
		p.pruneMatrix(prefix+"ffn1", layer.FFNWeights1.Data, currentTarget)
		p.pruneMatrix(prefix+"ffn2", layer.FFNWeights2.Data, currentTarget)
	}

	return nil
}

// pruneMatrix applies pruning to a weight matrix
func (p *Pruner) pruneMatrix(name string, data []float32, sparsity float64) {
	if len(data) == 0 {
		return
	}

	switch p.config.Method {
	case "magnitude":
		p.magnitudePruning(name, data, sparsity)
	case "structured":
		p.structuredPruning(name, data, sparsity)
	default:
		p.magnitudePruning(name, data, sparsity)
	}
}

// magnitudePruning applies magnitude-based pruning
func (p *Pruner) magnitudePruning(name string, data []float32, sparsity float64) {
	// Create or get existing mask
	mask, exists := p.masks[name]
	if !exists {
		mask = make([]bool, len(data))
		for i := range mask {
			mask[i] = true
		}
		p.masks[name] = mask
	}

	// Calculate threshold
	values := make([]float32, len(data))
	copy(values, data)
	for i := range values {
		values[i] = float32(math.Abs(float64(values[i])))
	}
	sort.Slice(values, func(i, j int) bool {
		return values[i] < values[j]
	})

	threshold := values[int(float64(len(values))*sparsity)]

	// Apply pruning
	numPruned := 0
	for i := range data {
		if float32(math.Abs(float64(data[i]))) <= threshold {
			data[i] = 0
			mask[i] = false
			numPruned++
		}
	}

	// Update statistics
	p.stats[name+"_sparsity"] = float64(numPruned) / float64(len(data))
}

// structuredPruning applies structured block pruning
func (p *Pruner) structuredPruning(name string, data []float32, sparsity float64) {
	blockSize := p.config.BlockSize
	if blockSize <= 0 {
		blockSize = 8 // Default block size
	}

	// Create or get existing mask
	mask, exists := p.masks[name]
	if !exists {
		mask = make([]bool, len(data))
		for i := range mask {
			mask[i] = true
		}
		p.masks[name] = mask
	}

	// Calculate block norms
	numBlocks := len(data) / blockSize
	if numBlocks == 0 {
		return
	}

	blockNorms := make([]struct {
		index int
		norm  float32
	}, numBlocks)

	for i := 0; i < numBlocks; i++ {
		start := i * blockSize
		end := start + blockSize
		if end > len(data) {
			end = len(data)
		}

		// Calculate L2 norm of block
		var norm float32
		for j := start; j < end; j++ {
			norm += data[j] * data[j]
		}
		norm = float32(math.Sqrt(float64(norm)))

		blockNorms[i] = struct {
			index int
			norm  float32
		}{i, norm}
	}

	// Sort blocks by norm
	sort.Slice(blockNorms, func(i, j int) bool {
		return blockNorms[i].norm < blockNorms[j].norm
	})

	// Prune blocks with lowest norms
	numBlocksToPrune := int(float64(numBlocks) * sparsity)
	numPruned := 0

	for i := 0; i < numBlocksToPrune; i++ {
		blockIdx := blockNorms[i].index
		start := blockIdx * blockSize
		end := start + blockSize
		if end > len(data) {
			end = len(data)
		}

		// Zero out block
		for j := start; j < end; j++ {
			data[j] = 0
			mask[j] = false
			numPruned++
		}
	}

	// Update statistics
	p.stats[name+"_sparsity"] = float64(numPruned) / float64(len(data))
}

// shouldDropLayer determines if a layer should be dropped
func (p *Pruner) shouldDropLayer() bool {
	if p.config.LayerDropout <= 0 {
		return false
	}
	return rand.Float64() < p.config.LayerDropout
}

// dropLayer zeros out all weights in a transformer layer
func (p *Pruner) dropLayer(layer *model.TransformerLayer) {
	zeroOut := func(data []float32) {
		for i := range data {
			data[i] = 0
		}
	}

	zeroOut(layer.QueryWeights.Data)
	zeroOut(layer.KeyWeights.Data)
	zeroOut(layer.ValueWeights.Data)
	zeroOut(layer.OutputWeights.Data)
	zeroOut(layer.FFNWeights1.Data)
	zeroOut(layer.FFNWeights2.Data)
}

// calculateCurrentSparsity calculates the current sparsity target
func (p *Pruner) calculateCurrentSparsity(step int) float64 {
	if p.config.GradualSteps <= 0 || step >= p.config.GradualSteps {
		return p.config.SparsityTarget
	}

	// Linear sparsity schedule
	progress := float64(step) / float64(p.config.GradualSteps)
	return p.config.SparsityTarget * progress
}

// GetPruningStats returns statistics about the pruning
func (p *Pruner) GetPruningStats() map[string]interface{} {
	stats := make(map[string]interface{})

	// Calculate average sparsity
	totalSparsity := 0.0
	numLayers := 0
	for name, sparsity := range p.stats {
		if strings.HasSuffix(name, "_sparsity") {
			totalSparsity += sparsity
			numLayers++
		}
	}

	stats["average_sparsity"] = totalSparsity / float64(numLayers)
	stats["method"] = p.config.Method
	stats["target_sparsity"] = p.config.SparsityTarget
	stats["block_size"] = p.config.BlockSize
	stats["layer_dropout"] = p.config.LayerDropout
	stats["per_layer_stats"] = p.stats

	return stats
}
