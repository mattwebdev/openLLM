package metrics

import (
	"strings"
	"unicode"
)

// METEORScore calculates the METEOR score for text generation
func METEORScore(reference, hypothesis string) float64 {
	// Split into words
	refWords := splitWords(reference)
	hypWords := splitWords(hypothesis)

	// Calculate unigram precision and recall
	precision := calculatePrecision(refWords, hypWords)
	recall := calculatePrecision(hypWords, refWords)

	// Calculate F1 score
	f1 := 2 * (precision * recall) / (precision + recall)

	// Calculate word order penalty
	penalty := calculateWordOrderPenalty(refWords, hypWords)

	// Final METEOR score
	return f1 * (1 - penalty)
}

// CIDErScore calculates the CIDEr score for image captioning
func CIDErScore(references []string, hypothesis string) float64 {
	// Calculate TF-IDF weights for n-grams
	refNgrams := make(map[string]float64)
	for _, ref := range references {
		ngrams := extractNgrams(ref)
		for _, ngram := range ngrams {
			refNgrams[ngram]++
		}
	}

	hypNgrams := extractNgrams(hypothesis)
	hypWeights := calculateTFIDF(hypNgrams, refNgrams, len(references))

	// Calculate cosine similarity
	similarity := calculateCosineSimilarity(hypWeights, refNgrams)

	return similarity
}

// HumanEvaluation represents metrics for human evaluation
type HumanEvaluation struct {
	Fluency     float64
	Relevance   float64
	Coherence   float64
	Creativity  float64
	Consistency float64
}

// HumanEvaluator manages human evaluations
type HumanEvaluator struct {
	evaluations []HumanEvaluation
}

// NewHumanEvaluator creates a new human evaluator
func NewHumanEvaluator() *HumanEvaluator {
	return &HumanEvaluator{}
}

// AddEvaluation adds a new human evaluation
func (e *HumanEvaluator) AddEvaluation(eval HumanEvaluation) {
	e.evaluations = append(e.evaluations, eval)
}

// GetAverageEvaluation returns the average of all human evaluations
func (e *HumanEvaluator) GetAverageEvaluation() HumanEvaluation {
	if len(e.evaluations) == 0 {
		return HumanEvaluation{}
	}

	var sum HumanEvaluation
	for _, eval := range e.evaluations {
		sum.Fluency += eval.Fluency
		sum.Relevance += eval.Relevance
		sum.Coherence += eval.Coherence
		sum.Creativity += eval.Creativity
		sum.Consistency += eval.Consistency
	}

	count := float64(len(e.evaluations))
	return HumanEvaluation{
		Fluency:     sum.Fluency / count,
		Relevance:   sum.Relevance / count,
		Coherence:   sum.Coherence / count,
		Creativity:  sum.Creativity / count,
		Consistency: sum.Consistency / count,
	}
}

// Helper functions
func calculatePrecision(a, b []string) float64 {
	matches := 0
	for _, word := range a {
		for _, other := range b {
			if word == other {
				matches++
				break
			}
		}
	}
	return float64(matches) / float64(len(a))
}

func calculateWordOrderPenalty(a, b []string) float64 {
	lcs := findLCS(a, b)
	return 1.0 - float64(lcs)/float64(max(len(a), len(b)))
}

func findLCS(a, b []string) int {
	m, n := len(a), len(b)
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}

	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if a[i-1] == b[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}
	}

	return dp[m][n]
}

func calculateTFIDF(ngrams []string, refNgrams map[string]float64, numRefs int) map[string]float64 {
	weights := make(map[string]float64)
	for _, ngram := range ngrams {
		tf := 1.0 // Term frequency
		idf := float64(numRefs) / (refNgrams[ngram] + 1)
		weights[ngram] = tf * idf
	}
	return weights
}

func calculateCosineSimilarity(a, b map[string]float64) float64 {
	var dotProduct, normA, normB float64

	for k, v := range a {
		dotProduct += v * b[k]
		normA += v * v
	}

	for _, v := range b {
		normB += v * v
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (normA * normB)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Helper functions for text processing
func splitWords(text string) []string {
	words := strings.FieldsFunc(text, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	return words
}

func extractNgrams(text string) []string {
	words := splitWords(text)
	ngrams := make([]string, 0, len(words))

	// Extract unigrams
	for _, word := range words {
		ngrams = append(ngrams, word)
	}

	// Extract bigrams
	for i := 0; i < len(words)-1; i++ {
		ngrams = append(ngrams, words[i]+" "+words[i+1])
	}

	// Extract trigrams
	for i := 0; i < len(words)-2; i++ {
		ngrams = append(ngrams, words[i]+" "+words[i+1]+" "+words[i+2])
	}

	return ngrams
}
