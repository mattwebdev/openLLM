package optimization

import (
	"testing"

	"openllm/pkg/model"
)

func TestNewMemoryOptimizer(t *testing.T) {
	config := MemoryConfig{
		EnableGradientCheckpointing: true,
		CheckpointEvery:             4,
		EnableMemoryEfficientAttn:   true,
		MaxActivationMemory:         1024 * 1024 * 1024, // 1GB
		MemoryBuffer:                0.1,
	}

	optimizer := NewMemoryOptimizer(config)

	if !optimizer.config.EnableGradientCheckpointing {
		t.Error("Expected gradient checkpointing to be enabled")
	}
	if !optimizer.config.EnableMemoryEfficientAttn {
		t.Error("Expected memory-efficient attention to be enabled")
	}
	if optimizer.config.MaxActivationMemory != 1024*1024*1024 {
		t.Error("Incorrect max activation memory")
	}
	if len(optimizer.checkpoints) != 0 {
		t.Error("Checkpoints map should be empty initially")
	}
}

func TestGradientCheckpointing(t *testing.T) {
	// Create test model
	modelConfig := &model.Config{
		VocabSize:    100,
		MaxSeqLength: 32,
		HiddenSize:   64,
		NumLayers:    8,
		NumHeads:     4,
		FFNDim:       128,
		Dropout:      0.1,
		LayerNormEps: 1e-5,
	}
	testModel := model.NewTransformer(modelConfig)

	config := MemoryConfig{
		EnableGradientCheckpointing: true,
		CheckpointEvery:             2,
		MaxActivationMemory:         1024 * 1024,
	}
	optimizer := NewMemoryOptimizer(config)

	// Apply optimizations
	err := optimizer.OptimizeMemory(testModel)
	if err != nil {
		t.Fatalf("Failed to optimize memory: %v", err)
	}

	// Check checkpoint locations
	stats := optimizer.GetMemoryStats()
	checkpointLayers := stats["checkpoint_layers"].([]int)
	expectedCheckpoints := []int{1, 3, 5, 7} // Every 2 layers in an 8-layer model

	if len(checkpointLayers) != len(expectedCheckpoints) {
		t.Errorf("Expected %d checkpoints, got %d", len(expectedCheckpoints), len(checkpointLayers))
	}
	for i, layer := range expectedCheckpoints {
		if checkpointLayers[i] != layer {
			t.Errorf("Expected checkpoint at layer %d, got %d", layer, checkpointLayers[i])
		}
	}
}

func TestMemoryEfficientAttention(t *testing.T) {
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

	config := MemoryConfig{
		EnableMemoryEfficientAttn: true,
		MaxActivationMemory:       1024 * 1024,
	}
	optimizer := NewMemoryOptimizer(config)

	// Apply optimizations
	err := optimizer.OptimizeMemory(testModel)
	if err != nil {
		t.Fatalf("Failed to optimize memory: %v", err)
	}

	// Check memory stats
	stats := optimizer.GetMemoryStats()
	if stats["memory_efficient_attn_enabled"].(bool) != true {
		t.Error("Memory-efficient attention should be enabled")
	}
	if stats["current_memory"].(int64) <= 0 {
		t.Error("Current memory should be positive")
	}
	if stats["peak_memory"].(int64) < stats["current_memory"].(int64) {
		t.Error("Peak memory should be at least current memory")
	}
}

func TestCheckpointSaveLoad(t *testing.T) {
	optimizer := NewMemoryOptimizer(MemoryConfig{
		EnableGradientCheckpointing: true,
		MaxActivationMemory:         1024 * 1024,
	})

	// Test saving checkpoint
	layerIdx := 0
	testState := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	optimizer.SaveCheckpoint(layerIdx, testState)

	// Test loading checkpoint
	loadedState, exists := optimizer.LoadCheckpoint(layerIdx)
	if !exists {
		t.Error("Expected checkpoint to exist")
	}
	if len(loadedState) != len(testState) {
		t.Errorf("Expected state length %d, got %d", len(testState), len(loadedState))
	}
	for i := range testState {
		if loadedState[i] != testState[i] {
			t.Errorf("State mismatch at index %d: expected %f, got %f", i, testState[i], loadedState[i])
		}
	}

	// Test clearing checkpoints
	optimizer.ClearCheckpoints()
	_, exists = optimizer.LoadCheckpoint(layerIdx)
	if exists {
		t.Error("Checkpoint should not exist after clearing")
	}
}

func TestMemoryStats(t *testing.T) {
	config := MemoryConfig{
		EnableGradientCheckpointing: true,
		EnableMemoryEfficientAttn:   true,
		MaxActivationMemory:         1024 * 1024,
		MemoryBuffer:                0.1,
	}
	optimizer := NewMemoryOptimizer(config)

	// Get stats
	stats := optimizer.GetMemoryStats()

	// Check required stats
	requiredStats := []string{
		"current_memory",
		"peak_memory",
		"memory_limit",
		"memory_utilization",
		"gradient_checkpointing_enabled",
		"memory_efficient_attn_enabled",
	}

	for _, stat := range requiredStats {
		if _, exists := stats[stat]; !exists {
			t.Errorf("Missing required stat: %s", stat)
		}
	}

	// Check stat values
	if stats["memory_limit"].(int64) != config.MaxActivationMemory {
		t.Error("Incorrect memory limit")
	}
	if stats["gradient_checkpointing_enabled"].(bool) != config.EnableGradientCheckpointing {
		t.Error("Incorrect gradient checkpointing enabled value")
	}
	if stats["memory_efficient_attn_enabled"].(bool) != config.EnableMemoryEfficientAttn {
		t.Error("Incorrect memory efficient attention enabled value")
	}
}

func TestMemoryOptimizationFailure(t *testing.T) {
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

	// Set very small memory limit
	config := MemoryConfig{
		EnableMemoryEfficientAttn: true,
		MaxActivationMemory:       100, // Too small
	}
	optimizer := NewMemoryOptimizer(config)

	// Optimization should fail due to insufficient memory
	err := optimizer.OptimizeMemory(testModel)
	if err == nil {
		t.Error("Expected optimization to fail with insufficient memory")
	}
}
