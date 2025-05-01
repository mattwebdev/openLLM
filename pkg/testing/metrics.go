package testing

import (
	"math"
	"strings"
)

// CoherenceMetrics calculates text coherence using various methods
type CoherenceMetrics struct {
	SentenceCount    int
	TransitionScore  float64
	ReferenceScore   float64
	TopicConsistency float64
	StructureScore   float64
}

// RelevanceMetrics calculates relevance between input and output
type RelevanceMetrics struct {
	KeywordOverlap     float64
	SemanticSimilarity float64
	TopicAlignment     float64
	ContextualMatch    float64
}

// FluencyMetrics calculates text fluency and naturalness
type FluencyMetrics struct {
	GrammarScore     float64
	ReadabilityScore float64
	NaturalFlow      float64
	Consistency      float64
}

// calculateCoherenceMetrics performs detailed coherence analysis
func calculateCoherenceMetrics(text string) CoherenceMetrics {
	sentences := splitIntoSentences(text)
	metrics := CoherenceMetrics{
		SentenceCount: len(sentences),
	}

	// Calculate transition score based on sentence connections
	metrics.TransitionScore = calculateTransitionScore(sentences)

	// Calculate reference consistency
	metrics.ReferenceScore = calculateReferenceScore(sentences)

	// Calculate topic consistency
	metrics.TopicConsistency = calculateTopicConsistency(sentences)

	// Calculate structural coherence
	metrics.StructureScore = calculateStructureScore(sentences)

	return metrics
}

// calculateRelevanceMetrics performs detailed relevance analysis
func calculateRelevanceMetrics(input, output string) RelevanceMetrics {
	metrics := RelevanceMetrics{}

	// Calculate keyword overlap
	metrics.KeywordOverlap = calculateKeywordOverlap(input, output)

	// Calculate semantic similarity
	metrics.SemanticSimilarity = calculateSemanticSimilarity(input, output)

	// Calculate topic alignment
	metrics.TopicAlignment = calculateTopicAlignment(input, output)

	// Calculate contextual match
	metrics.ContextualMatch = calculateContextualMatch(input, output)

	return metrics
}

// calculateFluencyMetrics performs detailed fluency analysis
func calculateFluencyMetrics(text string) FluencyMetrics {
	metrics := FluencyMetrics{}

	// Calculate grammar score
	metrics.GrammarScore = calculateGrammarScore(text)

	// Calculate readability
	metrics.ReadabilityScore = calculateReadabilityScore(text)

	// Calculate natural flow
	metrics.NaturalFlow = calculateNaturalFlow(text)

	// Calculate consistency
	metrics.Consistency = calculateConsistency(text)

	return metrics
}

// Helper functions for coherence metrics
func splitIntoSentences(text string) []string {
	// Simple sentence splitting for now
	return strings.Split(strings.ReplaceAll(text, ".", ". "), ". ")
}

func calculateTransitionScore(sentences []string) float64 {
	if len(sentences) < 2 {
		return 1.0
	}

	score := 0.0
	for i := 1; i < len(sentences); i++ {
		// Calculate word overlap between consecutive sentences
		prev := strings.Fields(strings.ToLower(sentences[i-1]))
		curr := strings.Fields(strings.ToLower(sentences[i]))
		overlap := calculateWordOverlap(prev, curr)
		score += overlap
	}
	return score / float64(len(sentences)-1)
}

func calculateReferenceScore(sentences []string) float64 {
	// TODO: Implement proper reference tracking
	return 0.8
}

func calculateTopicConsistency(sentences []string) float64 {
	// TODO: Implement topic modeling and consistency checking
	return 0.8
}

func calculateStructureScore(sentences []string) float64 {
	// TODO: Implement discourse structure analysis
	return 0.8
}

// Helper functions for relevance metrics
func calculateKeywordOverlap(input, output string) float64 {
	inputWords := strings.Fields(strings.ToLower(input))
	outputWords := strings.Fields(strings.ToLower(output))
	return calculateWordOverlap(inputWords, outputWords)
}

func calculateSemanticSimilarity(input, output string) float64 {
	// TODO: Implement proper semantic similarity using embeddings
	return 0.8
}

func calculateTopicAlignment(input, output string) float64 {
	// TODO: Implement topic modeling and alignment
	return 0.8
}

func calculateContextualMatch(input, output string) float64 {
	// TODO: Implement contextual analysis
	return 0.8
}

// Helper functions for fluency metrics
func calculateGrammarScore(text string) float64 {
	// TODO: Implement grammar checking
	return 0.8
}

func calculateReadabilityScore(text string) float64 {
	words := strings.Fields(text)
	sentences := splitIntoSentences(text)
	if len(sentences) == 0 {
		return 0.0
	}

	// Calculate Flesch Reading Ease score
	wordCount := float64(len(words))
	sentenceCount := float64(len(sentences))
	syllableCount := estimateSyllableCount(words)

	// Flesch Reading Ease = 206.835 - 1.015 × (words/sentences) - 84.6 × (syllables/words)
	score := 206.835 - 1.015*(wordCount/sentenceCount) - 84.6*(syllableCount/wordCount)
	return math.Max(0, math.Min(100, score)) / 100
}

func calculateNaturalFlow(text string) float64 {
	// TODO: Implement natural language flow analysis
	return 0.8
}

func calculateConsistency(text string) float64 {
	// TODO: Implement style and tone consistency analysis
	return 0.8
}

// Utility functions
func calculateWordOverlap(words1, words2 []string) float64 {
	if len(words1) == 0 || len(words2) == 0 {
		return 0.0
	}

	// Create word sets
	set1 := make(map[string]bool)
	for _, word := range words1 {
		set1[word] = true
	}

	// Count overlapping words
	overlap := 0
	for _, word := range words2 {
		if set1[word] {
			overlap++
		}
	}

	// Return Jaccard similarity
	union := float64(len(set1) + len(words2) - overlap)
	if union == 0 {
		return 0.0
	}
	return float64(overlap) / union
}

func estimateSyllableCount(words []string) float64 {
	total := 0.0
	for _, word := range words {
		total += float64(countSyllables(word))
	}
	return total
}

func countSyllables(word string) int {
	// Simple syllable counting heuristic
	count := 0
	prevIsVowel := false
	for _, c := range strings.ToLower(word) {
		isVowel := strings.ContainsRune("aeiouy", c)
		if isVowel && !prevIsVowel {
			count++
		}
		prevIsVowel = isVowel
	}
	if count == 0 {
		count = 1
	}
	return count
}
