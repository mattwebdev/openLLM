package preprocessing

import (
	"bufio"
	"os"
)

// DataLoader handles loading and processing training data
type DataLoader struct {
	preprocessor *Preprocessor
	vocabulary   *Vocabulary
}

// NewDataLoader creates a new data loader instance
func NewDataLoader(preprocessor *Preprocessor, vocabulary *Vocabulary) *DataLoader {
	return &DataLoader{
		preprocessor: preprocessor,
		vocabulary:   vocabulary,
	}
}

// LoadTextFile loads and processes a text file
func (dl *DataLoader) LoadTextFile(filename string) ([]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var texts []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		text := scanner.Text()
		processed, err := dl.preprocessor.PreprocessText(text)
		if err != nil {
			continue // Skip invalid texts
		}
		texts = append(texts, processed)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return texts, nil
}

// BuildVocabulary builds vocabulary from a list of texts
func (dl *DataLoader) BuildVocabulary(texts []string) {
	var allTokens []string
	for _, text := range texts {
		tokens := dl.preprocessor.Tokenize(text)
		allTokens = append(allTokens, tokens...)
	}
	dl.vocabulary.BuildFromTokens(allTokens)
}

// CreateTrainingPairs creates input-target pairs for training
func (dl *DataLoader) CreateTrainingPairs(texts []string, windowSize int) ([][]int, [][]int, error) {
	var inputs, targets [][]int

	for _, text := range texts {
		// Encode text to token IDs
		ids, err := dl.vocabulary.Encode(text)
		if err != nil {
			return nil, nil, err
		}

		// Create sliding window pairs
		for i := 0; i < len(ids)-windowSize; i++ {
			input := ids[i : i+windowSize]
			target := ids[i+windowSize : i+windowSize+1]

			inputs = append(inputs, input)
			targets = append(targets, target)
		}
	}

	return inputs, targets, nil
}

// SaveVocabulary saves the vocabulary to a file
func (dl *DataLoader) SaveVocabulary(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	for i := 0; i < dl.vocabulary.Size(); i++ {
		token, err := dl.vocabulary.GetToken(i)
		if err != nil {
			return err
		}
		_, err = file.WriteString(token + "\n")
		if err != nil {
			return err
		}
	}

	return nil
}

// LoadVocabulary loads a vocabulary from a file
func (dl *DataLoader) LoadVocabulary(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		token := scanner.Text()
		dl.vocabulary.AddToken(token)
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	return nil
}
