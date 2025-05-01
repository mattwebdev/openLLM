package optimization

import (
	"fmt"
	"sync"

	"openllm/pkg/model"
)

// InferenceConfig holds configuration for inference optimization
type InferenceConfig struct {
	EnableFusion    bool    // Enable kernel fusion
	EnableCaching   bool    // Enable KV caching
	BatchSize       int     // Batch size for inference
	MaxCacheSize    int     // Maximum cache size in elements
	CachePruneRatio float64 // Ratio of cache to prune when full
}

// InferenceOptimizer handles inference optimization
type InferenceOptimizer struct {
	config InferenceConfig
	cache  *KVCache
	stats  map[string]interface{}
	mu     sync.RWMutex
}

// KVCache implements caching for attention key/value pairs
type KVCache struct {
	keys    map[string][]float32
	values  map[string][]float32
	size    int
	maxSize int
}

// NewInferenceOptimizer creates a new inference optimizer
func NewInferenceOptimizer(config InferenceConfig) *InferenceOptimizer {
	cache := &KVCache{
		keys:    make(map[string][]float32),
		values:  make(map[string][]float32),
		maxSize: config.MaxCacheSize,
	}

	return &InferenceOptimizer{
		config: config,
		cache:  cache,
		stats:  make(map[string]interface{}),
	}
}

// OptimizeInference applies inference optimizations to the model
func (o *InferenceOptimizer) OptimizeInference(model *model.Transformer) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	if o.config.EnableFusion {
		if err := o.fuseOperations(model); err != nil {
			return fmt.Errorf("failed to fuse operations: %v", err)
		}
	}

	if o.config.EnableCaching {
		if err := o.setupCaching(model); err != nil {
			return fmt.Errorf("failed to setup caching: %v", err)
		}
	}

	return nil
}

// fuseOperations applies kernel fusion optimizations
func (o *InferenceOptimizer) fuseOperations(model *model.Transformer) error {
	// Fuse layer norm with attention
	for _, layer := range model.Layers {
		// Fuse Q/K/V projections
		o.fuseQKVProjection(layer)

		// Fuse attention with feedforward
		o.fuseAttentionFFN(layer)
	}

	o.stats["fused_operations"] = len(model.Layers) * 2 // QKV fusion and attention-FFN fusion
	return nil
}

// fuseQKVProjection fuses query, key, and value projections
func (o *InferenceOptimizer) fuseQKVProjection(layer *model.TransformerLayer) {
	// Calculate dimensions
	headDim := len(layer.QueryWeights.Data) / layer.NumHeads

	// Create combined Q/K/V weight matrix
	combinedSize := headDim * layer.NumHeads * 3 // 3 for Q, K, V
	qkvWeights := make([]float32, combinedSize)

	// Copy and reorganize weights for efficient batch computation
	for h := 0; h < layer.NumHeads; h++ {
		// Copy query weights
		qOffset := h * headDim
		copy(qkvWeights[qOffset:], layer.QueryWeights.Data[h*headDim:(h+1)*headDim])

		// Copy key weights
		kOffset := headDim*layer.NumHeads + h*headDim
		copy(qkvWeights[kOffset:], layer.KeyWeights.Data[h*headDim:(h+1)*headDim])

		// Copy value weights
		vOffset := 2*headDim*layer.NumHeads + h*headDim
		copy(qkvWeights[vOffset:], layer.ValueWeights.Data[h*headDim:(h+1)*headDim])
	}

	// Update layer weights
	layer.QueryWeights.Data = qkvWeights[:headDim*layer.NumHeads]
	layer.KeyWeights.Data = qkvWeights[headDim*layer.NumHeads : 2*headDim*layer.NumHeads]
	layer.ValueWeights.Data = qkvWeights[2*headDim*layer.NumHeads:]

	// Update statistics
	o.stats["qkv_fusion_size"] = combinedSize
	o.stats["qkv_fusion_heads"] = layer.NumHeads
	o.stats["qkv_fusion_head_dim"] = headDim
}

// fuseAttentionFFN fuses attention output with feedforward network
func (o *InferenceOptimizer) fuseAttentionFFN(layer *model.TransformerLayer) {
	// Calculate dimensions
	headDim := len(layer.QueryWeights.Data) / layer.NumHeads
	ffnDim := len(layer.FFNWeights1.Data) / layer.MaxSeqLength

	// Create fused weight matrix for attention output and first FFN layer
	fusedSize := headDim * layer.NumHeads * ffnDim
	fusedWeights := make([]float32, fusedSize)

	// Perform matrix multiplication for fusion
	for h := 0; h < layer.NumHeads; h++ {
		for i := 0; i < headDim; i++ {
			for j := 0; j < ffnDim; j++ {
				// Calculate fused weight
				attnIdx := h*headDim + i
				ffnIdx := i*ffnDim + j
				fusedIdx := h*headDim*ffnDim + i*ffnDim + j

				// Fuse attention output with FFN weights
				fusedWeights[fusedIdx] = layer.OutputWeights.Data[attnIdx] * layer.FFNWeights1.Data[ffnIdx]
			}
		}
	}

	// Update layer weights
	layer.OutputWeights.Data = fusedWeights
	layer.FFNWeights1.Data = nil // Mark as fused

	// Update statistics
	o.stats["attention_ffn_fusion_size"] = fusedSize
	o.stats["attention_ffn_fusion_heads"] = layer.NumHeads
	o.stats["attention_ffn_fusion_head_dim"] = headDim
	o.stats["attention_ffn_fusion_ffn_dim"] = ffnDim
}

// setupCaching initializes KV caching
func (o *InferenceOptimizer) setupCaching(model *model.Transformer) error {
	if o.cache.maxSize <= 0 {
		return fmt.Errorf("invalid cache size: %d", o.cache.maxSize)
	}

	// Clear existing cache
	o.cache.keys = make(map[string][]float32)
	o.cache.values = make(map[string][]float32)
	o.cache.size = 0

	return nil
}

// getCachedKV retrieves cached key/value pairs
func (o *InferenceOptimizer) getCachedKV(key string) ([]float32, []float32, bool) {
	o.mu.RLock()
	defer o.mu.RUnlock()

	k, existsK := o.cache.keys[key]
	v, existsV := o.cache.values[key]

	if existsK && existsV {
		return k, v, true
	}
	return nil, nil, false
}

// cacheKV stores key/value pairs in cache
func (o *InferenceOptimizer) cacheKV(key string, k, v []float32) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	// Check if cache is full
	if o.cache.size >= o.cache.maxSize {
		// Prune cache
		o.pruneCache()
	}

	// Add to cache
	o.cache.keys[key] = k
	o.cache.values[key] = v
	o.cache.size++

	return nil
}

// pruneCache removes least recently used entries
func (o *InferenceOptimizer) pruneCache() {
	targetSize := int(float64(o.cache.maxSize) * (1 - o.config.CachePruneRatio))

	// Simple LRU implementation - remove oldest entries
	// In a real implementation, we would use a proper LRU cache
	numToPrune := o.cache.size - targetSize
	pruned := 0

	for key := range o.cache.keys {
		delete(o.cache.keys, key)
		delete(o.cache.values, key)
		pruned++
		if pruned >= numToPrune {
			break
		}
	}

	o.cache.size = targetSize
}

// GetOptimizationStats returns statistics about the optimizations
func (o *InferenceOptimizer) GetOptimizationStats() map[string]interface{} {
	o.mu.RLock()
	defer o.mu.RUnlock()

	stats := make(map[string]interface{})
	for k, v := range o.stats {
		stats[k] = v
	}

	stats["cache_size"] = o.cache.size
	stats["cache_max_size"] = o.cache.maxSize
	stats["cache_utilization"] = float64(o.cache.size) / float64(o.cache.maxSize)
	stats["fusion_enabled"] = o.config.EnableFusion
	stats["caching_enabled"] = o.config.EnableCaching
	stats["batch_size"] = o.config.BatchSize

	return stats
}

// ClearCache clears the KV cache
func (o *InferenceOptimizer) ClearCache() {
	o.mu.Lock()
	defer o.mu.Unlock()

	o.cache.keys = make(map[string][]float32)
	o.cache.values = make(map[string][]float32)
	o.cache.size = 0
}
