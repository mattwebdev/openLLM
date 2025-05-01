package testing

import (
	"encoding/json"
	"testing"

	"openllm/pkg/model"
)

func TestNewTestSession(t *testing.T) {
	// Create a test model
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

	// Create test session
	session := NewTestSession(testModel)

	// Verify initial state
	if session.model != testModel {
		t.Error("Model not properly initialized")
	}
	if len(session.results) != 0 {
		t.Error("Results should be empty initially")
	}
	if len(session.metrics) != 0 {
		t.Error("Metrics should be empty initially")
	}
	if session.startTime.IsZero() {
		t.Error("Start time should be initialized")
	}
}

func TestRunTest(t *testing.T) {
	// Create test session
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
	session := NewTestSession(testModel)

	// Run a test
	result, err := session.RunTest("Test input")
	if err != nil {
		t.Fatalf("RunTest failed: %v", err)
	}

	// Verify result
	if result.Input != "Test input" {
		t.Error("Input not properly stored")
	}
	if result.Output == "" {
		t.Error("Output should not be empty")
	}
	if result.ProcessingTime <= 0 {
		t.Error("Processing time should be positive")
	}
	if result.Timestamp.IsZero() {
		t.Error("Timestamp should be set")
	}
	if result.Quality.Overall <= 0 || result.Quality.Overall > 1 {
		t.Error("Quality score should be between 0 and 1")
	}
}

func TestGetResults(t *testing.T) {
	// Create test session
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
	session := NewTestSession(testModel)

	// Run multiple tests
	inputs := []string{"Test 1", "Test 2", "Test 3"}
	for _, input := range inputs {
		_, err := session.RunTest(input)
		if err != nil {
			t.Fatalf("RunTest failed: %v", err)
		}
	}

	// Get results
	results := session.GetResults()

	// Verify results
	if len(results) != len(inputs) {
		t.Errorf("Expected %d results, got %d", len(inputs), len(results))
	}
	for i, result := range results {
		if result.Input != inputs[i] {
			t.Errorf("Result %d: expected input %s, got %s", i, inputs[i], result.Input)
		}
	}
}

func TestGetMetrics(t *testing.T) {
	// Create test session
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
	session := NewTestSession(testModel)

	// Run some tests
	for i := 0; i < 3; i++ {
		_, err := session.RunTest("Test input")
		if err != nil {
			t.Fatalf("RunTest failed: %v", err)
		}
	}

	// Get metrics
	metrics := session.GetMetrics()

	// Verify metrics
	requiredMetrics := []string{
		"total_tests",
		"total_tokens",
		"average_latency_ms",
		"average_perplexity",
		"average_quality_score",
		"session_duration_s",
		"tokens_per_second",
	}

	for _, metric := range requiredMetrics {
		if _, exists := metrics[metric]; !exists {
			t.Errorf("Missing required metric: %s", metric)
		}
	}

	if metrics["total_tests"].(int) != 3 {
		t.Error("Incorrect test count")
	}
}

func TestExportResults(t *testing.T) {
	// Create test session
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
	session := NewTestSession(testModel)

	// Run a test
	_, err := session.RunTest("Test input")
	if err != nil {
		t.Fatalf("RunTest failed: %v", err)
	}

	// Export results
	data, err := session.ExportResults()
	if err != nil {
		t.Fatalf("ExportResults failed: %v", err)
	}

	// Verify JSON structure
	var exported struct {
		Results []TestResult           `json:"results"`
		Metrics map[string]interface{} `json:"metrics"`
	}

	if err := json.Unmarshal(data, &exported); err != nil {
		t.Fatalf("Failed to parse exported JSON: %v", err)
	}

	if len(exported.Results) != 1 {
		t.Error("Expected 1 result in export")
	}
	if len(exported.Metrics) == 0 {
		t.Error("Expected metrics in export")
	}
}
