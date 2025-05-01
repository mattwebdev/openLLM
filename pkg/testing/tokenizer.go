package testing

import (
	"strings"
	"unicode"
)

// Tokenizer handles text tokenization and detokenization
type Tokenizer struct {
	vocab     map[string]int
	invVocab  map[int]string
	unkToken  int
	padToken  int
	eosToken  int
	bosToken  int
	maxTokens int
}

// NewTokenizer creates a new tokenizer instance
func NewTokenizer(maxTokens int) *Tokenizer {
	vocab := make(map[string]int)
	invVocab := make(map[int]string)

	// Add special tokens
	specialTokens := []string{"<unk>", "<pad>", "<eos>", "<bos>"}
	for i, token := range specialTokens {
		vocab[token] = i
		invVocab[i] = token
	}

	return &Tokenizer{
		vocab:     vocab,
		invVocab:  invVocab,
		unkToken:  0,
		padToken:  1,
		eosToken:  2,
		bosToken:  3,
		maxTokens: maxTokens,
	}
}

// AddToVocabulary adds a token to the vocabulary
func (t *Tokenizer) AddToVocabulary(token string) int {
	if id, exists := t.vocab[token]; exists {
		return id
	}
	id := len(t.vocab)
	t.vocab[token] = id
	t.invVocab[id] = token
	return id
}

// Tokenize converts text into token IDs
func (t *Tokenizer) Tokenize(text string) []int {
	// Preprocess text
	text = strings.TrimSpace(text)
	text = strings.ToLower(text)

	// Split into words and subwords
	tokens := make([]int, 0)
	tokens = append(tokens, t.bosToken)

	words := strings.FieldsFunc(text, func(r rune) bool {
		return unicode.IsSpace(r) || unicode.IsPunct(r)
	})

	for _, word := range words {
		// Simple word-level tokenization for now
		// TODO: Implement subword tokenization (BPE/WordPiece)
		if id, exists := t.vocab[word]; exists {
			tokens = append(tokens, id)
		} else {
			// Handle unknown tokens
			subTokens := t.tokenizeUnknown(word)
			tokens = append(tokens, subTokens...)
		}

		// Check max length
		if len(tokens) >= t.maxTokens-1 {
			break
		}
	}

	tokens = append(tokens, t.eosToken)

	// Pad if necessary
	if len(tokens) < t.maxTokens {
		padding := make([]int, t.maxTokens-len(tokens))
		for i := range padding {
			padding[i] = t.padToken
		}
		tokens = append(tokens, padding...)
	}

	return tokens[:t.maxTokens]
}

// Detokenize converts token IDs back to text
func (t *Tokenizer) Detokenize(tokens []int) string {
	var result strings.Builder

	for i, token := range tokens {
		// Skip special tokens
		if token == t.padToken || token == t.eosToken || token == t.bosToken {
			continue
		}

		// Get token text
		text, exists := t.invVocab[token]
		if !exists {
			text = t.invVocab[t.unkToken]
		}

		// Add space between tokens (except for punctuation)
		if i > 0 && !isPunctuation(text) {
			result.WriteString(" ")
		}

		result.WriteString(text)
	}

	return strings.TrimSpace(result.String())
}

// Helper functions

func (t *Tokenizer) tokenizeUnknown(word string) []int {
	// Simple character-level fallback for unknown words
	tokens := make([]int, 0)
	for _, c := range word {
		charToken := string(c)
		if id, exists := t.vocab[charToken]; exists {
			tokens = append(tokens, id)
		} else {
			tokens = append(tokens, t.unkToken)
		}
	}
	return tokens
}

func isPunctuation(s string) bool {
	if len(s) != 1 {
		return false
	}
	return unicode.IsPunct(rune(s[0]))
}

// BuildVocabulary builds vocabulary from a text corpus
func (t *Tokenizer) BuildVocabulary(texts []string) {
	wordFreq := make(map[string]int)

	// Count word frequencies
	for _, text := range texts {
		words := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
			return unicode.IsSpace(r) || unicode.IsPunct(r)
		})

		for _, word := range words {
			wordFreq[word]++
		}
	}

	// Add most frequent words to vocabulary
	for word := range wordFreq {
		t.AddToVocabulary(word)
	}
}

// GetVocabSize returns the current vocabulary size
func (t *Tokenizer) GetVocabSize() int {
	return len(t.vocab)
}

// IsSpecialToken checks if a token ID is a special token
func (t *Tokenizer) IsSpecialToken(id int) bool {
	return id == t.unkToken || id == t.padToken || id == t.eosToken || id == t.bosToken
}
