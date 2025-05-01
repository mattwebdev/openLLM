package preprocessing

import (
	"strings"
	"testing"
)

func TestPreprocessor(t *testing.T) {
	preprocessor := NewPreprocessor(1, 100, true, true)

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "basic text",
			input:    "Hello, World!",
			expected: "hello world",
		},
		{
			name:     "whitespace",
			input:    "  Hello   World  ",
			expected: "hello world",
		},
		{
			name:     "punctuation",
			input:    "Hello, World! How are you?",
			expected: "hello world how are you",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := preprocessor.PreprocessText(tt.input)
			if err != nil {
				t.Fatalf("PreprocessText() error = %v", err)
			}
			if result != tt.expected {
				t.Errorf("PreprocessText() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestVocabulary(t *testing.T) {
	vocab := NewVocabulary()

	// Test adding tokens
	vocab.AddToken("hello")
	vocab.AddToken("world")

	// Test getting IDs
	id, err := vocab.GetID("hello")
	if err != nil {
		t.Fatalf("GetID() error = %v", err)
	}
	if id != 0 {
		t.Errorf("GetID() = %v, want %v", id, 0)
	}

	id, err = vocab.GetID("world")
	if err != nil {
		t.Fatalf("GetID() error = %v", err)
	}
	if id != 1 {
		t.Errorf("GetID() = %v, want %v", id, 1)
	}

	// Test getting tokens
	token, err := vocab.GetToken(0)
	if err != nil {
		t.Fatalf("GetToken() error = %v", err)
	}
	if token != "hello" {
		t.Errorf("GetToken() = %v, want %v", token, "hello")
	}

	token, err = vocab.GetToken(1)
	if err != nil {
		t.Fatalf("GetToken() error = %v", err)
	}
	if token != "world" {
		t.Errorf("GetToken() = %v, want %v", token, "world")
	}

	// Test encoding
	encoded, err := vocab.Encode("hello world")
	if err != nil {
		t.Fatalf("Encode() error = %v", err)
	}
	expected := []int{0, 1}
	if len(encoded) != len(expected) {
		t.Errorf("Encode() length = %v, want %v", len(encoded), len(expected))
	}
	for i, v := range encoded {
		if v != expected[i] {
			t.Errorf("Encode()[%d] = %v, want %v", i, v, expected[i])
		}
	}

	// Test decoding
	decoded, err := vocab.Decode(encoded)
	if err != nil {
		t.Fatalf("Decode() error = %v", err)
	}
	if decoded != "hello world" {
		t.Errorf("Decode() = %v, want %v", decoded, "hello world")
	}
}

func TestDataLoader(t *testing.T) {
	preprocessor := NewPreprocessor(1, 100, true, true)
	vocab := NewVocabulary()
	loader := NewDataLoader(preprocessor, vocab)

	// Test loading text file
	texts, err := loader.LoadTextFile("testdata/sample.txt")
	if err != nil {
		t.Fatalf("LoadTextFile() error = %v", err)
	}
	if len(texts) == 0 {
		t.Error("LoadTextFile() returned empty slice")
	}

	// Test building vocabulary
	loader.BuildVocabulary(texts)
	if vocab.Size() == 0 {
		t.Error("BuildVocabulary() did not add any tokens")
	}

	// Test creating training pairs
	inputs, targets, err := loader.CreateTrainingPairs(texts, 3)
	if err != nil {
		t.Fatalf("CreateTrainingPairs() error = %v", err)
	}
	if len(inputs) == 0 || len(targets) == 0 {
		t.Error("CreateTrainingPairs() returned empty slices")
	}
}

func TestAugmenter(t *testing.T) {
	augmenter := NewAugmenter(0.1, 0.1, 0.1, 0.1)
	augmenter.AddSynonyms("happy", []string{"joyful", "cheerful"})
	augmenter.AddSynonyms("sad", []string{"unhappy", "depressed"})

	// Test word dropout
	text := "I am happy today"
	augmented := augmenter.wordDropout(strings.Fields(text))
	if len(augmented) > len(strings.Fields(text)) {
		t.Error("wordDropout() added words instead of removing them")
	}

	// Test word swap
	words := []string{"I", "am", "happy"}
	swapped := augmenter.wordSwap(words)
	if len(swapped) != len(words) {
		t.Error("wordSwap() changed the number of words")
	}

	// Test word insertion
	inserted := augmenter.wordInsert(words)
	if len(inserted) < len(words) {
		t.Error("wordInsert() removed words instead of adding them")
	}

	// Test word replacement
	replaced := augmenter.wordReplace(words)
	if len(replaced) != len(words) {
		t.Error("wordReplace() changed the number of words")
	}

	// Test back translation
	backTranslated := augmenter.BackTranslation(text)
	if backTranslated == text {
		t.Error("BackTranslation() did not modify the text")
	}

	// Test random insertion
	vocab := []string{"hello", "world"}
	randomInserted := augmenter.RandomInsertion(text, vocab)
	if randomInserted == text {
		t.Error("RandomInsertion() did not modify the text")
	}
}
