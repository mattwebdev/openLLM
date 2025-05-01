package testing

import (
	"strings"
	"testing"
)

func TestNewTokenizer(t *testing.T) {
	tokenizer := NewTokenizer(512)

	// Test initial state
	if tokenizer.GetVocabSize() != 4 {
		t.Errorf("Expected 4 special tokens, got %d", tokenizer.GetVocabSize())
	}

	// Test special tokens
	specialTokens := []int{tokenizer.unkToken, tokenizer.padToken, tokenizer.eosToken, tokenizer.bosToken}
	for _, token := range specialTokens {
		if !tokenizer.IsSpecialToken(token) {
			t.Errorf("Token %d should be special", token)
		}
	}
}

func TestTokenization(t *testing.T) {
	tokenizer := NewTokenizer(10)

	// Add some words to vocabulary
	words := []string{"hello", "world", "test", "tokenization"}
	for _, word := range words {
		tokenizer.AddToVocabulary(word)
	}

	tests := []struct {
		name  string
		input string
		want  int // Expected number of tokens (including special tokens)
	}{
		{
			name:  "Simple text",
			input: "hello world",
			want:  10, // BOS + 2 words + EOS + padding
		},
		{
			name:  "Unknown words",
			input: "unknown words here",
			want:  10, // BOS + unknown tokens + EOS + padding
		},
		{
			name:  "Empty text",
			input: "",
			want:  10, // BOS + EOS + padding
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokens := tokenizer.Tokenize(tt.input)
			if len(tokens) != tt.want {
				t.Errorf("Tokenize() got %v tokens, want %v", len(tokens), tt.want)
			}
			if tokens[0] != tokenizer.bosToken {
				t.Error("First token should be BOS")
			}
		})
	}
}

func TestDetokenization(t *testing.T) {
	tokenizer := NewTokenizer(512)

	// Add test vocabulary
	words := []string{"hello", "world", "test", "tokenization"}
	for _, word := range words {
		tokenizer.AddToVocabulary(word)
	}

	tests := []struct {
		name  string
		input string
	}{
		{
			name:  "Known words",
			input: "hello world test",
		},
		{
			name:  "Mixed known/unknown",
			input: "hello unknown test",
		},
		{
			name:  "Punctuation",
			input: "hello, world!",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokens := tokenizer.Tokenize(tt.input)
			output := tokenizer.Detokenize(tokens)

			// The output might not match exactly due to tokenization/detokenization,
			// but it should contain the main words
			for _, word := range words {
				if strings.Contains(tt.input, word) && !strings.Contains(output, word) {
					t.Errorf("Detokenize() missing word %q in output %q", word, output)
				}
			}
		})
	}
}

func TestVocabularyBuilding(t *testing.T) {
	tokenizer := NewTokenizer(512)
	corpus := []string{
		"hello world",
		"test tokenization",
		"hello test world",
	}

	// Build vocabulary
	tokenizer.BuildVocabulary(corpus)

	// Test vocabulary size (4 special tokens + 4 unique words)
	expectedSize := 8
	if size := tokenizer.GetVocabSize(); size != expectedSize {
		t.Errorf("Expected vocabulary size %d, got %d", expectedSize, size)
	}

	// Test tokenization with built vocabulary
	tokens := tokenizer.Tokenize("hello world test")
	if len(tokens) != 512 {
		t.Errorf("Expected 512 tokens (with padding), got %d", len(tokens))
	}
}

func TestMaxLength(t *testing.T) {
	maxLength := 5
	tokenizer := NewTokenizer(maxLength)

	// Add some words
	words := []string{"a", "b", "c", "d", "e", "f", "g"}
	for _, word := range words {
		tokenizer.AddToVocabulary(word)
	}

	// Test long input
	input := "a b c d e f g"
	tokens := tokenizer.Tokenize(input)

	if len(tokens) != maxLength {
		t.Errorf("Expected token length %d, got %d", maxLength, len(tokens))
	}
}
