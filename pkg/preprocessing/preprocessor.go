package preprocessing

import (
	"errors"
	"strings"
	"unicode"
)

// Preprocessor handles data preprocessing tasks
type Preprocessor struct {
	// Configuration
	maxSequenceLength int
	vocabSize         int
	lowercase         bool
	removePunctuation bool
}

// NewPreprocessor creates a new preprocessor instance
func NewPreprocessor(maxSequenceLength, vocabSize int, lowercase, removePunctuation bool) *Preprocessor {
	return &Preprocessor{
		maxSequenceLength: maxSequenceLength,
		vocabSize:         vocabSize,
		lowercase:         lowercase,
		removePunctuation: removePunctuation,
	}
}

// PreprocessText performs basic text preprocessing
func (p *Preprocessor) PreprocessText(text string) (string, error) {
	if len(text) == 0 {
		return "", errors.New("empty text input")
	}

	// Convert to lowercase if configured
	if p.lowercase {
		text = strings.ToLower(text)
	}

	// Remove punctuation if configured
	if p.removePunctuation {
		text = p.removePunctuationFromText(text)
	}

	// Trim whitespace
	text = strings.TrimSpace(text)

	// Truncate if exceeds max sequence length
	if len(text) > p.maxSequenceLength {
		text = text[:p.maxSequenceLength]
	}

	return text, nil
}

// removePunctuationFromText removes punctuation from text
func (p *Preprocessor) removePunctuationFromText(text string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsPunct(r) {
			return -1
		}
		return r
	}, text)
}

// Tokenize splits text into tokens
func (p *Preprocessor) Tokenize(text string) []string {
	// Basic tokenization by whitespace
	// TODO: Implement more sophisticated tokenization
	return strings.Fields(text)
}

// ValidateText performs basic text validation
func (p *Preprocessor) ValidateText(text string) error {
	if len(text) == 0 {
		return errors.New("text is empty")
	}

	if len(text) > p.maxSequenceLength {
		return errors.New("text exceeds maximum sequence length")
	}

	return nil
}
