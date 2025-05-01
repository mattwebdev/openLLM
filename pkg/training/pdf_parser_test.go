package training

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPDFParser(t *testing.T) {
	// Create temporary directory
	tempDir, err := os.MkdirTemp("", "pdf-parser-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create a simple PDF file for testing
	pdfContent := `%PDF-1.7
1 0 obj
<< /Type /Catalog
   /Pages 2 0 R
>>
endobj

2 0 obj
<< /Type /Pages
   /Kids [3 0 R]
   /Count 1
>>
endobj

3 0 obj
<< /Type /Page
   /Parent 2 0 R
   /MediaBox [0 0 612 792]
   /Contents 4 0 R
   /Resources << /Font << /F1 5 0 R >> >>
>>
endobj

4 0 obj
<< /Length 68 >>
stream
BT
/F1 12 Tf
72 720 Td
(This is a test PDF file for the OpenLLM project.) Tj
ET
endstream
endobj

5 0 obj
<< /Type /Font
   /Subtype /Type1
   /BaseFont /Helvetica
>>
endobj

xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000241 00000 n
0000000361 00000 n
trailer
<< /Size 6
   /Root 1 0 R
>>
startxref
432
%%EOF`

	pdfPath := filepath.Join(tempDir, "test.pdf")
	if err := os.WriteFile(pdfPath, []byte(pdfContent), 0644); err != nil {
		t.Fatalf("Failed to create test PDF file: %v", err)
	}

	// Create parser
	parser := NewPDFParser(0) // Process all pages

	// Test ExtractText
	text, err := parser.ExtractText(pdfPath)
	if err != nil {
		t.Fatalf("Failed to extract text: %v", err)
	}

	expectedText := "This is a test PDF file for the OpenLLM project."
	if !strings.Contains(text, expectedText) {
		t.Errorf("Expected text to contain %q, got %q", expectedText, text)
	}

	// Test ExtractTextFromReader
	f, err := os.Open(pdfPath)
	if err != nil {
		t.Fatalf("Failed to open PDF file: %v", err)
	}
	defer f.Close()

	fileInfo, err := f.Stat()
	if err != nil {
		t.Fatalf("Failed to get file info: %v", err)
	}

	text, err = parser.ExtractTextFromReader(f, fileInfo.Size())
	if err != nil {
		t.Fatalf("Failed to extract text from reader: %v", err)
	}

	if !strings.Contains(text, expectedText) {
		t.Errorf("Expected text to contain %q, got %q", expectedText, text)
	}

	// Test with maxPages limit
	parser = NewPDFParser(1)
	text, err = parser.ExtractText(pdfPath)
	if err != nil {
		t.Fatalf("Failed to extract text with page limit: %v", err)
	}

	if !strings.Contains(text, expectedText) {
		t.Errorf("Expected text to contain %q, got %q", expectedText, text)
	}

	// Test CleanText
	dirtyText := "This\n\nis\n\na\n\ntest"
	cleanText := parser.CleanText(dirtyText)
	if cleanText != dirtyText {
		t.Errorf("Expected cleaned text %q, got %q", dirtyText, cleanText)
	}
}

func TestPDFParserErrors(t *testing.T) {
	parser := NewPDFParser(0)

	// Test with non-existent file
	_, err := parser.ExtractText("nonexistent.pdf")
	if err == nil {
		t.Error("Expected error for non-existent file")
	}

	// Test with invalid PDF content
	invalidContent := []byte("This is not a PDF file")
	invalidPDF := bytes.NewReader(invalidContent)
	_, err = parser.ExtractTextFromReader(invalidPDF, int64(len(invalidContent)))
	if err == nil {
		t.Error("Expected error for invalid PDF content")
	}
}
