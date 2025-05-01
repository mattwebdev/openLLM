package testing

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"openllm/pkg/model"
)

// TestResult represents the result of a single test
type TestResult struct {
	Input           string    `json:"input"`
	Output          string    `json:"output"`
	ProcessingTime  float64   `json:"processing_time_ms"`
	TokenCount      int       `json:"token_count"`
	Timestamp       time.Time `json:"timestamp"`
	PerplexityScore float64   `json:"perplexity_score"`
	Quality         struct {
		Coherence float64 `json:"coherence"`
		Relevance float64 `json:"relevance"`
		Fluency   float64 `json:"fluency"`
		Overall   float64 `json:"overall"`
	} `json:"quality"`
}

// TestSession manages a model testing session
type TestSession struct {
	model       *model.Transformer
	tokenizer   *Tokenizer
	results     []TestResult
	mu          sync.RWMutex
	metrics     map[string]float64
	startTime   time.Time
	totalTokens int
}

// NewTestSession creates a new testing session
func NewTestSession(model *model.Transformer) *TestSession {
	return &TestSession{
		model:     model,
		tokenizer: NewTokenizer(model.Config.MaxSeqLength),
		results:   make([]TestResult, 0),
		metrics:   make(map[string]float64),
		startTime: time.Now(),
	}
}

// RunTest performs a single test with the given input
func (ts *TestSession) RunTest(input string) (*TestResult, error) {
	start := time.Now()

	// Tokenize input
	tokens := ts.tokenizer.Tokenize(input)

	// Generate output
	outputProbs, err := ts.model.Forward(tokens)
	if err != nil {
		return nil, fmt.Errorf("model inference failed: %v", err)
	}

	// Convert probabilities to token IDs (simple argmax)
	outputTokens := make([]int, len(outputProbs)/ts.model.Config.VocabSize)
	for i := range outputTokens {
		startIdx := i * ts.model.Config.VocabSize
		maxProb := float32(-1)
		maxIdx := 0
		for j := 0; j < ts.model.Config.VocabSize; j++ {
			if outputProbs[startIdx+j] > maxProb {
				maxProb = outputProbs[startIdx+j]
				maxIdx = j
			}
		}
		outputTokens[i] = maxIdx
	}

	// Detokenize output
	outputText := ts.tokenizer.Detokenize(outputTokens)

	// Calculate metrics
	result := &TestResult{
		Input:          input,
		Output:         outputText,
		ProcessingTime: float64(time.Since(start).Milliseconds()),
		TokenCount:     len(tokens),
		Timestamp:      time.Now(),
	}

	// Calculate detailed quality metrics
	coherenceMetrics := calculateCoherenceMetrics(outputText)
	relevanceMetrics := calculateRelevanceMetrics(input, outputText)
	fluencyMetrics := calculateFluencyMetrics(outputText)

	// Aggregate quality scores
	result.Quality.Coherence = (coherenceMetrics.TransitionScore +
		coherenceMetrics.ReferenceScore +
		coherenceMetrics.TopicConsistency +
		coherenceMetrics.StructureScore) / 4.0

	result.Quality.Relevance = (relevanceMetrics.KeywordOverlap +
		relevanceMetrics.SemanticSimilarity +
		relevanceMetrics.TopicAlignment +
		relevanceMetrics.ContextualMatch) / 4.0

	result.Quality.Fluency = (fluencyMetrics.GrammarScore +
		fluencyMetrics.ReadabilityScore +
		fluencyMetrics.NaturalFlow +
		fluencyMetrics.Consistency) / 4.0

	result.Quality.Overall = (result.Quality.Coherence +
		result.Quality.Relevance +
		result.Quality.Fluency) / 3.0

	// Calculate perplexity
	result.PerplexityScore = calculatePerplexity(outputText)

	// Store result
	ts.mu.Lock()
	ts.results = append(ts.results, *result)
	ts.totalTokens += result.TokenCount
	ts.updateMetrics(result)
	ts.mu.Unlock()

	return result, nil
}

// GetResults returns all test results
func (ts *TestSession) GetResults() []TestResult {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	results := make([]TestResult, len(ts.results))
	copy(results, ts.results)
	return results
}

// GetMetrics returns current testing metrics
func (ts *TestSession) GetMetrics() map[string]interface{} {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	return map[string]interface{}{
		"total_tests":           len(ts.results),
		"total_tokens":          ts.totalTokens,
		"average_latency_ms":    ts.metrics["avg_latency"],
		"average_perplexity":    ts.metrics["avg_perplexity"],
		"average_quality_score": ts.metrics["avg_quality"],
		"session_duration_s":    time.Since(ts.startTime).Seconds(),
		"tokens_per_second":     float64(ts.totalTokens) / time.Since(ts.startTime).Seconds(),
	}
}

// ExportResults exports test results to JSON
func (ts *TestSession) ExportResults() ([]byte, error) {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	return json.Marshal(struct {
		Results []TestResult           `json:"results"`
		Metrics map[string]interface{} `json:"metrics"`
	}{
		Results: ts.results,
		Metrics: ts.GetMetrics(),
	})
}

// Helper functions for quality metrics

func calculateCoherence(input, output string) float64 {
	// TODO: Implement proper coherence calculation using NLP techniques
	// For now, return a placeholder value
	return 0.8
}

func calculateRelevance(input, output string) float64 {
	// TODO: Implement proper relevance calculation using semantic similarity
	// For now, return a placeholder value
	return 0.8
}

func calculateFluency(output string) float64 {
	// TODO: Implement proper fluency calculation using language model scoring
	// For now, return a placeholder value
	return 0.8
}

func calculatePerplexity(output string) float64 {
	// TODO: Implement proper perplexity calculation
	// For now, return a placeholder value
	return 1.0
}

func (ts *TestSession) updateMetrics(result *TestResult) {
	// Update running averages
	n := float64(len(ts.results))
	ts.metrics["avg_latency"] = (ts.metrics["avg_latency"]*(n-1) + result.ProcessingTime) / n
	ts.metrics["avg_perplexity"] = (ts.metrics["avg_perplexity"]*(n-1) + result.PerplexityScore) / n
	ts.metrics["avg_quality"] = (ts.metrics["avg_quality"]*(n-1) + result.Quality.Overall) / n
}
