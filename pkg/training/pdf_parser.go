package training

import (
	"bytes"
	"fmt"
	"io"
	"os"

	"github.com/ledongthuc/pdf"
)

// PDFParser handles extraction of text from PDF files
type PDFParser struct {
	maxPages int // Maximum number of pages to process, 0 for all pages
}

// NewPDFParser creates a new PDF parser
func NewPDFParser(maxPages int) *PDFParser {
	return &PDFParser{
		maxPages: maxPages,
	}
}

// ExtractText extracts text content from a PDF file
func (p *PDFParser) ExtractText(filepath string) (string, error) {
	// Open the PDF file
	f, err := os.Open(filepath)
	if err != nil {
		return "", fmt.Errorf("failed to open PDF file: %v", err)
	}
	defer f.Close()

	// Get file size
	fileInfo, err := f.Stat()
	if err != nil {
		return "", fmt.Errorf("failed to get file info: %v", err)
	}

	// Read the PDF
	reader, err := pdf.NewReader(f, fileInfo.Size())
	if err != nil {
		return "", fmt.Errorf("failed to create PDF reader: %v", err)
	}

	var buf bytes.Buffer
	numPages := reader.NumPage()

	// If maxPages is set, limit the number of pages to process
	if p.maxPages > 0 && p.maxPages < numPages {
		numPages = p.maxPages
	}

	// Process each page
	for pageNum := 1; pageNum <= numPages; pageNum++ {
		page := reader.Page(pageNum)
		if page.V.IsNull() {
			continue
		}

		// Extract plain text
		text, err := page.GetPlainText(nil)
		if err != nil {
			return "", fmt.Errorf("failed to extract text from page %d: %v", pageNum, err)
		}

		// Write to buffer with page separator
		if pageNum > 1 {
			buf.WriteString("\n\n")
		}
		buf.WriteString(text)
	}

	return buf.String(), nil
}

// ExtractTextFromReader extracts text content from an io.ReaderAt containing PDF data
func (p *PDFParser) ExtractTextFromReader(r io.ReaderAt, size int64) (string, error) {
	// Create PDF reader from io.ReaderAt
	reader, err := pdf.NewReader(r, size)
	if err != nil {
		return "", fmt.Errorf("failed to create PDF reader: %v", err)
	}

	var buf bytes.Buffer
	numPages := reader.NumPage()

	// If maxPages is set, limit the number of pages to process
	if p.maxPages > 0 && p.maxPages < numPages {
		numPages = p.maxPages
	}

	// Process each page
	for pageNum := 1; pageNum <= numPages; pageNum++ {
		page := reader.Page(pageNum)
		if page.V.IsNull() {
			continue
		}

		// Extract plain text
		text, err := page.GetPlainText(nil)
		if err != nil {
			return "", fmt.Errorf("failed to extract text from page %d: %v", pageNum, err)
		}

		// Write to buffer with page separator
		if pageNum > 1 {
			buf.WriteString("\n\n")
		}
		buf.WriteString(text)
	}

	return buf.String(), nil
}

// CleanText performs basic cleaning of extracted PDF text
func (p *PDFParser) CleanText(text string) string {
	// TODO: Implement more sophisticated text cleaning
	// For now, just handle basic cleanup
	text = bytes.NewBufferString(text).String()
	return text
}
