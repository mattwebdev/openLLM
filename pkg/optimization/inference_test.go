package optimization

import (
	"fmt"
	"testing"

	"openllm/pkg/model"
)

func TestNewInferenceOptimizer(t *testing.T) {
	config := InferenceConfig{
		EnableFusion:    true,
		EnableCaching:   true,
		BatchSize:       32,
		MaxCacheSize:    1000,
		CachePruneRatio: 0.2,
	}

	optimizer := NewInferenceOptimizer(config)

	if !optimizer.config.EnableFusion {
		t.Error("Expected fusion to be enabled")
	}
	if !optimizer.config.EnableCaching {
		t.Error("Expected caching to be enabled")
	}
	if optimizer.config.BatchSize != 32 {
		t.Errorf("Expected batch size 32, got %d", optimizer.config.BatchSize)
	}
	if optimizer.cache.maxSize != 1000 {
		t.Errorf("Expected max cache size 1000, got %d", optimizer.cache.maxSize)
	}
}

func TestKVCache(t *testing.T) {
	config := InferenceConfig{
		EnableCaching:   true,
		MaxCacheSize:    10,
		CachePruneRatio: 0.5,
	}
	optimizer := NewInferenceOptimizer(config)

	// Test cache operations
	testKey := "test_key"
	testKeys := []float32{1.0, 2.0, 3.0}
	testValues := []float32{4.0, 5.0, 6.0}

	// Test caching
	err := optimizer.cacheKV(testKey, testKeys, testValues)
	if err != nil {
		t.Fatalf("Failed to cache KV: %v", err)
	}

	// Test retrieval
	k, v, exists := optimizer.getCachedKV(testKey)
	if !exists {
		t.Error("Expected cached KV to exist")
	}
	if len(k) != len(testKeys) || len(v) != len(testValues) {
		t.Error("Cached KV size mismatch")
	}
	for i := range testKeys {
		if k[i] != testKeys[i] || v[i] != testValues[i] {
			t.Error("Cached KV value mismatch")
		}
	}

	// Test cache pruning
	for i := 0; i < 15; i++ {
		key := fmt.Sprintf("key_%d", i)
		optimizer.cacheKV(key, testKeys, testValues)
	}

	stats := optimizer.GetOptimizationStats()
	if stats["cache_size"].(int) > config.MaxCacheSize {
		t.Error("Cache exceeded maximum size")
	}
}

func TestFusionOptimization(t *testing.T) {
	// Create test model
	modelConfig := &model.Config{
		VocabSize:    100,
		MaxSeqLength: 32,
		HiddenSize:   64,
		NumLayers:    2,
		NumHeads:     4,
		FFNDim:       128,
		Dropout:      0.1,
		LayerNormEps: 1e-5,
	}
	testModel := model.NewTransformer(modelConfig)

	// Create optimizer
	config := InferenceConfig{
		EnableFusion:  true,
		EnableCaching: false,
		BatchSize:     32,
	}
	optimizer := NewInferenceOptimizer(config)

	// Apply optimizations
	err := optimizer.OptimizeInference(testModel)
	if err != nil {
		t.Fatalf("Failed to optimize inference: %v", err)
	}

	// Check fusion results
	stats := optimizer.GetOptimizationStats()
	expectedFusions := len(testModel.Layers) * 2 // QKV fusion and attention-FFN fusion
	if stats["fused_operations"].(int) != expectedFusions {
		t.Errorf("Expected %d fused operations, got %d", expectedFusions, stats["fused_operations"].(int))
	}

	// Check layer modifications
	for i, layer := range testModel.Layers {
		// Check QKV fusion
		totalQKVSize := len(layer.QueryWeights.Data) + len(layer.KeyWeights.Data) + len(layer.ValueWeights.Data)
		if totalQKVSize == 0 {
			t.Errorf("Layer %d: QKV weights missing after fusion", i)
		}

		// Check attention-FFN fusion
		if layer.FFNWeights1.Data != nil {
			t.Errorf("Layer %d: FFN weights should be nil after fusion", i)
		}
		if len(layer.OutputWeights.Data) == 0 {
			t.Errorf("Layer %d: Output weights missing after fusion", i)
		}
	}
}

func TestOptimizationStats(t *testing.T) {
	config := InferenceConfig{
		EnableFusion:    true,
		EnableCaching:   true,
		BatchSize:       32,
		MaxCacheSize:    1000,
		CachePruneRatio: 0.2,
	}
	optimizer := NewInferenceOptimizer(config)

	// Get initial stats
	stats := optimizer.GetOptimizationStats()

	// Check required stats
	requiredStats := []string{
		"cache_size",
		"cache_max_size",
		"cache_utilization",
		"fusion_enabled",
		"caching_enabled",
		"batch_size",
	}

	for _, stat := range requiredStats {
		if _, exists := stats[stat]; !exists {
			t.Errorf("Missing required stat: %s", stat)
		}
	}

	// Check stat values
	if stats["fusion_enabled"].(bool) != config.EnableFusion {
		t.Error("Incorrect fusion_enabled value")
	}
	if stats["caching_enabled"].(bool) != config.EnableCaching {
		t.Error("Incorrect caching_enabled value")
	}
	if stats["batch_size"].(int) != config.BatchSize {
		t.Error("Incorrect batch_size value")
	}
}

func TestCachePruning(t *testing.T) {
	config := InferenceConfig{
		EnableCaching:   true,
		MaxCacheSize:    10,
		CachePruneRatio: 0.5,
	}
	optimizer := NewInferenceOptimizer(config)

	// Fill cache
	testData := []float32{1.0, 2.0, 3.0}
	for i := 0; i < config.MaxCacheSize*2; i++ {
		key := fmt.Sprintf("key_%d", i)
		optimizer.cacheKV(key, testData, testData)
	}

	// Check cache size
	stats := optimizer.GetOptimizationStats()
	if stats["cache_size"].(int) > config.MaxCacheSize {
		t.Errorf("Cache size %d exceeds max size %d", stats["cache_size"].(int), config.MaxCacheSize)
	}

	// Check pruning ratio
	expectedSize := int(float64(config.MaxCacheSize) * (1 - config.CachePruneRatio))
	if stats["cache_size"].(int) != expectedSize {
		t.Errorf("Expected cache size %d after pruning, got %d", expectedSize, stats["cache_size"].(int))
	}
}

func TestClearCache(t *testing.T) {
	config := InferenceConfig{
		EnableCaching: true,
		MaxCacheSize:  100,
	}
	optimizer := NewInferenceOptimizer(config)

	// Add some items to cache
	testData := []float32{1.0, 2.0, 3.0}
	for i := 0; i < 10; i++ {
		key := fmt.Sprintf("key_%d", i)
		optimizer.cacheKV(key, testData, testData)
	}

	// Clear cache
	optimizer.ClearCache()

	// Check cache is empty
	stats := optimizer.GetOptimizationStats()
	if stats["cache_size"].(int) != 0 {
		t.Errorf("Expected empty cache, got size %d", stats["cache_size"].(int))
	}
	if stats["cache_utilization"].(float64) != 0 {
		t.Error("Expected zero cache utilization after clear")
	}
}
