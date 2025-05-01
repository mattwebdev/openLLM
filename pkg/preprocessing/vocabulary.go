package preprocessing

import (
	"errors"
	"sort"
	"strings"
)

// Vocabulary handles token-to-id mapping
type Vocabulary struct {
	tokenToID map[string]int
	idToToken map[int]string
	size      int
}

// NewVocabulary creates a new vocabulary instance
func NewVocabulary() *Vocabulary {
	return &Vocabulary{
		tokenToID: make(map[string]int),
		idToToken: make(map[int]string),
		size:      0,
	}
}

// AddToken adds a token to the vocabulary
func (v *Vocabulary) AddToken(token string) int {
	if id, exists := v.tokenToID[token]; exists {
		return id
	}

	id := v.size
	v.tokenToID[token] = id
	v.idToToken[id] = token
	v.size++
	return id
}

// GetID returns the ID for a token
func (v *Vocabulary) GetID(token string) (int, error) {
	if id, exists := v.tokenToID[token]; exists {
		return id, nil
	}
	return 0, errors.New("token not found in vocabulary")
}

// GetToken returns the token for an ID
func (v *Vocabulary) GetToken(id int) (string, error) {
	if token, exists := v.idToToken[id]; exists {
		return token, nil
	}
	return "", errors.New("id not found in vocabulary")
}

// Size returns the vocabulary size
func (v *Vocabulary) Size() int {
	return v.size
}

// BuildFromTokens builds vocabulary from a list of tokens
func (v *Vocabulary) BuildFromTokens(tokens []string) {
	// Count token frequencies
	freq := make(map[string]int)
	for _, token := range tokens {
		freq[token]++
	}

	// Sort tokens by frequency
	type tokenFreq struct {
		token string
		freq  int
	}
	var sorted []tokenFreq
	for token, count := range freq {
		sorted = append(sorted, tokenFreq{token, count})
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].freq > sorted[j].freq
	})

	// Add tokens to vocabulary
	for _, tf := range sorted {
		v.AddToken(tf.token)
	}
}

// Encode converts text to token IDs
func (v *Vocabulary) Encode(text string) ([]int, error) {
	tokens := strings.Fields(text)
	ids := make([]int, len(tokens))
	for i, token := range tokens {
		id, err := v.GetID(token)
		if err != nil {
			return nil, err
		}
		ids[i] = id
	}
	return ids, nil
}

// Decode converts token IDs to text
func (v *Vocabulary) Decode(ids []int) (string, error) {
	tokens := make([]string, len(ids))
	for i, id := range ids {
		token, err := v.GetToken(id)
		if err != nil {
			return "", err
		}
		tokens[i] = token
	}
	return strings.Join(tokens, " "), nil
}
