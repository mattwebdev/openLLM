package logging

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// LogLevel represents the severity level of a log message
type LogLevel int

const (
	Debug LogLevel = iota
	Info
	Warning
	Error
)

// LogEntry represents a single log entry
type LogEntry struct {
	Timestamp time.Time
	Level     LogLevel
	Message   string
	Data      map[string]interface{}
}

// Logger handles logging operations
type Logger struct {
	mu       sync.Mutex
	file     *os.File
	minLevel LogLevel
	entries  []LogEntry
	flushDur time.Duration
}

// NewLogger creates a new logger instance
func NewLogger(logFile string, minLevel LogLevel) (*Logger, error) {
	// Create log directory if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(logFile), 0755); err != nil {
		return nil, fmt.Errorf("failed to create log directory: %v", err)
	}

	file, err := os.OpenFile(logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open log file: %v", err)
	}

	logger := &Logger{
		file:     file,
		minLevel: minLevel,
		flushDur: time.Minute,
	}

	// Start periodic flushing
	go logger.periodicFlush()

	return logger, nil
}

// Debug logs a debug message
func (l *Logger) Debug(msg string, data map[string]interface{}) {
	l.log(Debug, msg, data)
}

// Info logs an info message
func (l *Logger) Info(msg string, data map[string]interface{}) {
	l.log(Info, msg, data)
}

// Warning logs a warning message
func (l *Logger) Warning(msg string, data map[string]interface{}) {
	l.log(Warning, msg, data)
}

// Error logs an error message
func (l *Logger) Error(msg string, data map[string]interface{}) {
	l.log(Error, msg, data)
}

// log handles the actual logging process
func (l *Logger) log(level LogLevel, msg string, data map[string]interface{}) {
	if level < l.minLevel {
		return
	}

	entry := LogEntry{
		Timestamp: time.Now(),
		Level:     level,
		Message:   msg,
		Data:      data,
	}

	l.mu.Lock()
	l.entries = append(l.entries, entry)
	l.mu.Unlock()

	// Also write to stdout/stderr
	if level >= Error {
		fmt.Fprintf(os.Stderr, "[%s] %s\n", level.String(), msg)
	} else {
		fmt.Fprintf(os.Stdout, "[%s] %s\n", level.String(), msg)
	}
}

// GetEntries returns all log entries
func (l *Logger) GetEntries() []LogEntry {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.entries
}

// GetEntriesByLevel returns log entries filtered by level
func (l *Logger) GetEntriesByLevel(level LogLevel) []LogEntry {
	l.mu.Lock()
	defer l.mu.Unlock()

	var filtered []LogEntry
	for _, entry := range l.entries {
		if entry.Level == level {
			filtered = append(filtered, entry)
		}
	}
	return filtered
}

// GetEntriesInRange returns log entries within a time range
func (l *Logger) GetEntriesInRange(start, end time.Time) []LogEntry {
	l.mu.Lock()
	defer l.mu.Unlock()

	var filtered []LogEntry
	for _, entry := range l.entries {
		if !entry.Timestamp.Before(start) && !entry.Timestamp.After(end) {
			filtered = append(filtered, entry)
		}
	}
	return filtered
}

// Close closes the logger and its associated file
func (l *Logger) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if err := l.flush(); err != nil {
		return err
	}
	return l.file.Close()
}

// flush writes the current entries to the log file
func (l *Logger) flush() error {
	if len(l.entries) == 0 {
		return nil
	}

	encoder := json.NewEncoder(l.file)
	for _, entry := range l.entries {
		if err := encoder.Encode(entry); err != nil {
			return err
		}
	}

	l.entries = nil
	return nil
}

// periodicFlush periodically flushes the logs to disk
func (l *Logger) periodicFlush() {
	ticker := time.NewTicker(l.flushDur)
	defer ticker.Stop()

	for range ticker.C {
		if err := l.flush(); err != nil {
			// Log error but continue
			continue
		}
	}
}

// String returns the string representation of a log level
func (l LogLevel) String() string {
	switch l {
	case Debug:
		return "DEBUG"
	case Info:
		return "INFO"
	case Warning:
		return "WARNING"
	case Error:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}
