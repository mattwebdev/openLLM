package preprocessing

import (
	"regexp"
	"strings"
	"unicode"
)

// QualityChecker handles data quality assessment
type QualityChecker struct {
	minLength        int
	maxLength        int
	minWordCount     int
	maxWordCount     int
	allowedChars     *regexp.Regexp
	excludedWords    map[string]bool
	excludedPatterns []*regexp.Regexp
}

// NewQualityChecker creates a new quality checker
func NewQualityChecker(minLength, maxLength, minWordCount, maxWordCount int) *QualityChecker {
	return &QualityChecker{
		minLength:     minLength,
		maxLength:     maxLength,
		minWordCount:  minWordCount,
		maxWordCount:  maxWordCount,
		allowedChars:  regexp.MustCompile(`^[a-zA-Z0-9\s.,!?;:'"-]+$`),
		excludedWords: make(map[string]bool),
		excludedPatterns: []*regexp.Regexp{
			regexp.MustCompile(`http[s]?://\S+`), // URLs
			regexp.MustCompile(`@\w+`),           // Mentions
			regexp.MustCompile(`#\w+`),           // Hashtags
		},
	}
}

// CheckTextQuality performs quality checks on text
func (q *QualityChecker) CheckTextQuality(text string) (bool, []string) {
	var issues []string

	// Check length
	if len(text) < q.minLength {
		issues = append(issues, "text too short")
	}
	if len(text) > q.maxLength {
		issues = append(issues, "text too long")
	}

	// Check word count
	words := strings.Fields(text)
	if len(words) < q.minWordCount {
		issues = append(issues, "insufficient word count")
	}
	if len(words) > q.maxWordCount {
		issues = append(issues, "excessive word count")
	}

	// Check character set
	if !q.allowedChars.MatchString(text) {
		issues = append(issues, "contains invalid characters")
	}

	// Check for excluded words
	for _, word := range words {
		if q.excludedWords[strings.ToLower(word)] {
			issues = append(issues, "contains excluded word: "+word)
		}
	}

	// Check for excluded patterns
	for _, pattern := range q.excludedPatterns {
		if pattern.MatchString(text) {
			issues = append(issues, "contains excluded pattern")
		}
	}

	// Check for repeated characters
	if q.hasRepeatedChars(text) {
		issues = append(issues, "contains repeated characters")
	}

	// Check for proper capitalization
	if !q.hasProperCapitalization(text) {
		issues = append(issues, "improper capitalization")
	}

	return len(issues) == 0, issues
}

// hasRepeatedChars checks for excessive character repetition
func (q *QualityChecker) hasRepeatedChars(text string) bool {
	if len(text) < 3 {
		return false
	}

	runes := []rune(text)
	for i := 0; i < len(runes)-2; i++ {
		if runes[i] == runes[i+1] && runes[i] == runes[i+2] {
			return true
		}
	}
	return false
}

// hasProperCapitalization checks for proper sentence capitalization
func (q *QualityChecker) hasProperCapitalization(text string) bool {
	if len(text) == 0 {
		return true
	}

	// Check first character
	if !unicode.IsUpper(rune(text[0])) {
		return false
	}

	// Check after periods
	periods := strings.Split(text, ".")
	for i := 1; i < len(periods); i++ {
		period := strings.TrimSpace(periods[i])
		if len(period) > 0 && !unicode.IsUpper(rune(period[0])) {
			return false
		}
	}

	return true
}

// AddExcludedWord adds a word to the excluded words list
func (q *QualityChecker) AddExcludedWord(word string) {
	q.excludedWords[strings.ToLower(word)] = true
}

// AddExcludedPattern adds a pattern to the excluded patterns list
func (q *QualityChecker) AddExcludedPattern(pattern string) error {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return err
	}
	q.excludedPatterns = append(q.excludedPatterns, re)
	return nil
}
