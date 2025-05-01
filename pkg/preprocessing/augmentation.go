package preprocessing

import (
	"math/rand"
	"strings"
	"time"
)

// Augmenter handles data augmentation
type Augmenter struct {
	wordDropProb    float64
	wordSwapProb    float64
	wordInsertProb  float64
	wordReplaceProb float64
	synonyms        map[string][]string
}

// NewAugmenter creates a new augmenter
func NewAugmenter(wordDropProb, wordSwapProb, wordInsertProb, wordReplaceProb float64) *Augmenter {
	rand.Seed(time.Now().UnixNano())
	return &Augmenter{
		wordDropProb:    wordDropProb,
		wordSwapProb:    wordSwapProb,
		wordInsertProb:  wordInsertProb,
		wordReplaceProb: wordReplaceProb,
		synonyms:        make(map[string][]string),
	}
}

// AddSynonyms adds synonyms for a word
func (a *Augmenter) AddSynonyms(word string, synonyms []string) {
	a.synonyms[word] = synonyms
}

// AugmentText performs data augmentation on text
func (a *Augmenter) AugmentText(text string) string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return text
	}

	// Word dropout
	words = a.wordDropout(words)

	// Word swap
	words = a.wordSwap(words)

	// Word insertion
	words = a.wordInsert(words)

	// Word replacement
	words = a.wordReplace(words)

	return strings.Join(words, " ")
}

// wordDropout randomly drops words
func (a *Augmenter) wordDropout(words []string) []string {
	var result []string
	for _, word := range words {
		if rand.Float64() > a.wordDropProb {
			result = append(result, word)
		}
	}
	return result
}

// wordSwap randomly swaps adjacent words
func (a *Augmenter) wordSwap(words []string) []string {
	if len(words) < 2 {
		return words
	}

	result := make([]string, len(words))
	copy(result, words)

	for i := 0; i < len(words)-1; i++ {
		if rand.Float64() < a.wordSwapProb {
			result[i], result[i+1] = result[i+1], result[i]
		}
	}

	return result
}

// wordInsert randomly inserts synonyms
func (a *Augmenter) wordInsert(words []string) []string {
	if len(words) == 0 {
		return words
	}

	var result []string
	for _, word := range words {
		result = append(result, word)
		if rand.Float64() < a.wordInsertProb {
			if synonyms, exists := a.synonyms[word]; exists && len(synonyms) > 0 {
				// Insert a random synonym
				result = append(result, synonyms[rand.Intn(len(synonyms))])
			}
		}
	}

	return result
}

// wordReplace randomly replaces words with synonyms
func (a *Augmenter) wordReplace(words []string) []string {
	result := make([]string, len(words))
	copy(result, words)

	for i, word := range words {
		if rand.Float64() < a.wordReplaceProb {
			if synonyms, exists := a.synonyms[word]; exists && len(synonyms) > 0 {
				// Replace with a random synonym
				result[i] = synonyms[rand.Intn(len(synonyms))]
			}
		}
	}

	return result
}

// BackTranslation simulates back translation
func (a *Augmenter) BackTranslation(text string) string {
	// This is a simplified version that just performs some word replacements
	// In a real implementation, this would use actual translation APIs
	words := strings.Fields(text)
	for i, word := range words {
		if synonyms, exists := a.synonyms[word]; exists && len(synonyms) > 0 {
			words[i] = synonyms[rand.Intn(len(synonyms))]
		}
	}
	return strings.Join(words, " ")
}

// RandomInsertion randomly inserts words from the vocabulary
func (a *Augmenter) RandomInsertion(text string, vocab []string) string {
	if len(vocab) == 0 {
		return text
	}

	words := strings.Fields(text)
	if len(words) == 0 {
		return text
	}

	// Insert random words at random positions
	for i := 0; i < len(words); i++ {
		if rand.Float64() < a.wordInsertProb {
			randomWord := vocab[rand.Intn(len(vocab))]
			words = append(words[:i], append([]string{randomWord}, words[i:]...)...)
			i++ // Skip the inserted word
		}
	}

	return strings.Join(words, " ")
}
