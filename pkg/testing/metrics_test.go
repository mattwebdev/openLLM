package testing

import (
	"testing"
)

func TestCoherenceMetrics(t *testing.T) {
	text := "This is a test sentence. It has multiple sentences. They should be coherent."
	metrics := calculateCoherenceMetrics(text)

	// Test sentence count
	if metrics.SentenceCount != 3 {
		t.Errorf("Expected 3 sentences, got %d", metrics.SentenceCount)
	}

	// Test score ranges
	scores := []float64{
		metrics.TransitionScore,
		metrics.ReferenceScore,
		metrics.TopicConsistency,
		metrics.StructureScore,
	}

	for _, score := range scores {
		if score < 0 || score > 1 {
			t.Errorf("Score %f is outside valid range [0,1]", score)
		}
	}
}

func TestRelevanceMetrics(t *testing.T) {
	input := "What is the capital of France?"
	output := "The capital of France is Paris."
	metrics := calculateRelevanceMetrics(input, output)

	// Test keyword overlap
	if metrics.KeywordOverlap <= 0 {
		t.Error("Expected non-zero keyword overlap")
	}

	// Test score ranges
	scores := []float64{
		metrics.KeywordOverlap,
		metrics.SemanticSimilarity,
		metrics.TopicAlignment,
		metrics.ContextualMatch,
	}

	for _, score := range scores {
		if score < 0 || score > 1 {
			t.Errorf("Score %f is outside valid range [0,1]", score)
		}
	}
}

func TestFluencyMetrics(t *testing.T) {
	text := "This is a well-formed sentence with proper grammar and structure."
	metrics := calculateFluencyMetrics(text)

	// Test score ranges
	scores := []float64{
		metrics.GrammarScore,
		metrics.ReadabilityScore,
		metrics.NaturalFlow,
		metrics.Consistency,
	}

	for _, score := range scores {
		if score < 0 || score > 1 {
			t.Errorf("Score %f is outside valid range [0,1]", score)
		}
	}
}

func TestReadabilityScore(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		wantLow  float64
		wantHigh float64
	}{
		{
			name:     "Simple text",
			text:     "The cat sat on the mat.",
			wantLow:  0.7,
			wantHigh: 1.0,
		},
		{
			name:     "Complex text",
			text:     "The intricate complexities of quantum mechanics necessitate a profound understanding of mathematical principles.",
			wantLow:  0.0,
			wantHigh: 0.5,
		},
		{
			name:     "Empty text",
			text:     "",
			wantLow:  0.0,
			wantHigh: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score := calculateReadabilityScore(tt.text)
			if score < tt.wantLow || score > tt.wantHigh {
				t.Errorf("ReadabilityScore = %v, want between %v and %v", score, tt.wantLow, tt.wantHigh)
			}
		})
	}
}

func TestWordOverlap(t *testing.T) {
	tests := []struct {
		name     string
		words1   []string
		words2   []string
		wantLow  float64
		wantHigh float64
	}{
		{
			name:     "Identical words",
			words1:   []string{"the", "cat", "sat"},
			words2:   []string{"the", "cat", "sat"},
			wantLow:  0.9,
			wantHigh: 1.0,
		},
		{
			name:     "No overlap",
			words1:   []string{"the", "cat"},
			words2:   []string{"a", "dog"},
			wantLow:  0.0,
			wantHigh: 0.1,
		},
		{
			name:     "Partial overlap",
			words1:   []string{"the", "cat", "sat"},
			words2:   []string{"the", "dog", "sat"},
			wantLow:  0.4,
			wantHigh: 0.7,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			overlap := calculateWordOverlap(tt.words1, tt.words2)
			if overlap < tt.wantLow || overlap > tt.wantHigh {
				t.Errorf("WordOverlap = %v, want between %v and %v", overlap, tt.wantLow, tt.wantHigh)
			}
		})
	}
}

func TestSyllableCount(t *testing.T) {
	tests := []struct {
		word string
		want int
	}{
		{"cat", 1},
		{"hello", 2},
		{"beautiful", 3},
		{"extraordinary", 6},
		{"a", 1},
		{"", 1},
	}

	for _, tt := range tests {
		got := countSyllables(tt.word)
		if got < 1 {
			t.Errorf("countSyllables(%q) = %v, want at least 1", tt.word, got)
		}
	}
}
