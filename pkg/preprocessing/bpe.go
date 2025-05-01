package preprocessing

import (
	"strings"
)

// BPETokenizer implements Byte Pair Encoding
type BPETokenizer struct {
	merges      map[string]string
	vocab       map[string]int
	vocabSize   int
	maxTokenLen int
}

// NewBPETokenizer creates a new BPE tokenizer
func NewBPETokenizer(vocabSize int) *BPETokenizer {
	return &BPETokenizer{
		merges:      make(map[string]string),
		vocab:       make(map[string]int),
		vocabSize:   vocabSize,
		maxTokenLen: 0,
	}
}

// Train trains the BPE tokenizer on a corpus
func (b *BPETokenizer) Train(corpus []string) {
	// Initialize vocabulary with characters
	vocab := make(map[string]int)
	for _, text := range corpus {
		for _, char := range text {
			vocab[string(char)]++
		}
	}

	// Initialize pairs
	pairs := make(map[string]int)
	for _, text := range corpus {
		chars := strings.Split(text, "")
		for i := 0; i < len(chars)-1; i++ {
			pair := chars[i] + chars[i+1]
			pairs[pair]++
		}
	}

	// Perform BPE merges
	for len(vocab) < b.vocabSize {
		// Find most frequent pair
		var bestPair string
		maxFreq := 0
		for pair, freq := range pairs {
			if freq > maxFreq {
				bestPair = pair
				maxFreq = freq
			}
		}

		if maxFreq == 0 {
			break
		}

		// Add merge to vocabulary
		b.merges[bestPair] = bestPair
		vocab[bestPair] = maxFreq

		// Update pairs
		newPairs := make(map[string]int)
		for _, text := range corpus {
			chars := strings.Split(text, "")
			i := 0
			for i < len(chars)-1 {
				if chars[i]+chars[i+1] == bestPair {
					// Merge the pair
					chars[i] = bestPair
					chars = append(chars[:i+1], chars[i+2:]...)
				} else {
					i++
				}
			}
			// Update pairs for the new sequence
			for i := 0; i < len(chars)-1; i++ {
				pair := chars[i] + chars[i+1]
				newPairs[pair]++
			}
		}
		pairs = newPairs
	}

	// Build final vocabulary
	b.vocab = vocab
}

// Tokenize tokenizes text using BPE
func (b *BPETokenizer) Tokenize(text string) []string {
	// Split into characters
	tokens := strings.Split(text, "")

	// Apply merges
	for {
		bestPair := ""
		bestPos := -1

		// Find the first mergeable pair
		for i := 0; i < len(tokens)-1; i++ {
			pair := tokens[i] + tokens[i+1]
			if _, exists := b.merges[pair]; exists {
				bestPair = pair
				bestPos = i
				break
			}
		}

		if bestPos == -1 {
			break
		}

		// Merge the pair
		tokens[bestPos] = bestPair
		tokens = append(tokens[:bestPos+1], tokens[bestPos+2:]...)
	}

	return tokens
}

// Encode encodes text into token IDs
func (b *BPETokenizer) Encode(text string) []int {
	tokens := b.Tokenize(text)
	ids := make([]int, len(tokens))
	for i, token := range tokens {
		ids[i] = b.vocab[token]
	}
	return ids
}

// Decode decodes token IDs into text
func (b *BPETokenizer) Decode(ids []int) string {
	// Create reverse vocabulary
	revVocab := make(map[int]string)
	for token, id := range b.vocab {
		revVocab[id] = token
	}

	// Decode tokens
	tokens := make([]string, len(ids))
	for i, id := range ids {
		tokens[i] = revVocab[id]
	}

	return strings.Join(tokens, "")
}
