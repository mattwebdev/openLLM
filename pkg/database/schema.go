package database

import (
	"database/sql"
	"fmt"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// DB represents the database connection
type DB struct {
	*sql.DB
}

// Paper represents a training paper in the database
type Paper struct {
	ID           int64
	Filename     string
	FilePath     string
	FileType     string // "pdf", "md", or "rss"
	Status       string // "pending", "processing", "completed", "error"
	ErrorMessage string
	AddedAt      time.Time
	ProcessedAt  *time.Time
	TokenCount   int
	Hash         string // SHA256 hash to prevent duplicates
	Content      string // Full content of the article/paper
}

// TrainingMetric represents a training metric record
type TrainingMetric struct {
	ID           int64
	Timestamp    time.Time
	Epoch        int
	Step         int
	Loss         float64
	LearningRate float64
	PaperID      int64
	CheckpointID int64
}

// Checkpoint represents a saved model checkpoint
type Checkpoint struct {
	ID        int64
	Path      string
	Timestamp time.Time
	Epoch     int
	Step      int
	Metrics   string // JSON string of metrics
}

// NewDB creates a new database connection
func NewDB(path string) (*DB, error) {
	db, err := sql.Open("sqlite3", path)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %v", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %v", err)
	}

	return &DB{db}, nil
}

// Initialize creates the database tables
func (db *DB) Initialize() error {
	// Create papers table
	_, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS papers (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			filename TEXT NOT NULL,
			file_path TEXT NOT NULL,
			file_type TEXT NOT NULL,
			status TEXT NOT NULL,
			error_message TEXT,
			added_at TIMESTAMP NOT NULL,
			processed_at TIMESTAMP,
			token_count INTEGER DEFAULT 0,
			hash TEXT NOT NULL UNIQUE,
			content TEXT
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create papers table: %v", err)
	}

	// Create training_metrics table
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS training_metrics (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp TIMESTAMP NOT NULL,
			epoch INTEGER NOT NULL,
			step INTEGER NOT NULL,
			loss REAL NOT NULL,
			learning_rate REAL NOT NULL,
			paper_id INTEGER,
			checkpoint_id INTEGER,
			FOREIGN KEY (paper_id) REFERENCES papers (id),
			FOREIGN KEY (checkpoint_id) REFERENCES checkpoints (id)
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create training_metrics table: %v", err)
	}

	// Create checkpoints table
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS checkpoints (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			path TEXT NOT NULL,
			timestamp TIMESTAMP NOT NULL,
			epoch INTEGER NOT NULL,
			step INTEGER NOT NULL,
			metrics TEXT NOT NULL
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create checkpoints table: %v", err)
	}

	return nil
}

// Close closes the database connection
func (db *DB) Close() error {
	return db.DB.Close()
}
