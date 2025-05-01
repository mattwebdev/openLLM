package database

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"time"
)

// AddPaper adds a new paper to the database
func (db *DB) AddPaper(filename, filepath, fileType string) (*Paper, error) {
	// Calculate file hash
	hash, err := calculateFileHash(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate file hash: %v", err)
	}

	// Check for duplicates
	var exists bool
	err = db.QueryRow("SELECT EXISTS(SELECT 1 FROM papers WHERE hash = ?)", hash).Scan(&exists)
	if err != nil {
		return nil, fmt.Errorf("failed to check for duplicates: %v", err)
	}
	if exists {
		return nil, fmt.Errorf("paper with hash %s already exists", hash)
	}

	// Insert paper
	result, err := db.Exec(`
		INSERT INTO papers (filename, file_path, file_type, status, added_at, hash)
		VALUES (?, ?, ?, ?, ?, ?)
	`, filename, filepath, fileType, "pending", time.Now(), hash)
	if err != nil {
		return nil, fmt.Errorf("failed to insert paper: %v", err)
	}

	id, err := result.LastInsertId()
	if err != nil {
		return nil, fmt.Errorf("failed to get last insert id: %v", err)
	}

	return &Paper{
		ID:       id,
		Filename: filename,
		FilePath: filepath,
		FileType: fileType,
		Status:   "pending",
		AddedAt:  time.Now(),
		Hash:     hash,
	}, nil
}

// UpdatePaperStatus updates the status of a paper
func (db *DB) UpdatePaperStatus(id int64, status string, errorMsg string) error {
	var processedAt *time.Time
	if status == "completed" || status == "error" {
		now := time.Now()
		processedAt = &now
	}

	_, err := db.Exec(`
		UPDATE papers 
		SET status = ?, error_message = ?, processed_at = ?
		WHERE id = ?
	`, status, errorMsg, processedAt, id)

	if err != nil {
		return fmt.Errorf("failed to update paper status: %v", err)
	}
	return nil
}

// AddTrainingMetric adds a new training metric record
func (db *DB) AddTrainingMetric(metric *TrainingMetric) error {
	_, err := db.Exec(`
		INSERT INTO training_metrics (
			timestamp, epoch, step, loss, learning_rate, paper_id, checkpoint_id
		) VALUES (?, ?, ?, ?, ?, ?, ?)
	`, metric.Timestamp, metric.Epoch, metric.Step, metric.Loss,
		metric.LearningRate, metric.PaperID, metric.CheckpointID)

	if err != nil {
		return fmt.Errorf("failed to insert training metric: %v", err)
	}
	return nil
}

// AddCheckpoint adds a new checkpoint record
func (db *DB) AddCheckpoint(checkpoint *Checkpoint) error {
	metricsJSON, err := json.Marshal(checkpoint.Metrics)
	if err != nil {
		return fmt.Errorf("failed to marshal metrics: %v", err)
	}

	_, err = db.Exec(`
		INSERT INTO checkpoints (path, timestamp, epoch, step, metrics)
		VALUES (?, ?, ?, ?, ?)
	`, checkpoint.Path, checkpoint.Timestamp, checkpoint.Epoch,
		checkpoint.Step, string(metricsJSON))

	if err != nil {
		return fmt.Errorf("failed to insert checkpoint: %v", err)
	}
	return nil
}

// GetPendingPapers returns all papers with "pending" status
func (db *DB) GetPendingPapers() ([]Paper, error) {
	rows, err := db.Query(`
		SELECT id, filename, file_path, file_type, status, error_message, 
			   added_at, processed_at, token_count, hash
		FROM papers 
		WHERE status = 'pending'
		ORDER BY added_at ASC
	`)
	if err != nil {
		return nil, fmt.Errorf("failed to query pending papers: %v", err)
	}
	defer rows.Close()

	var papers []Paper
	for rows.Next() {
		var p Paper
		err := rows.Scan(
			&p.ID, &p.Filename, &p.FilePath, &p.FileType, &p.Status,
			&p.ErrorMessage, &p.AddedAt, &p.ProcessedAt, &p.TokenCount, &p.Hash,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan paper row: %v", err)
		}
		papers = append(papers, p)
	}

	return papers, nil
}

// GetRecentMetrics returns recent training metrics
func (db *DB) GetRecentMetrics(limit int) ([]TrainingMetric, error) {
	rows, err := db.Query(`
		SELECT id, timestamp, epoch, step, loss, learning_rate, paper_id, checkpoint_id
		FROM training_metrics
		ORDER BY timestamp DESC
		LIMIT ?
	`, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query recent metrics: %v", err)
	}
	defer rows.Close()

	var metrics []TrainingMetric
	for rows.Next() {
		var m TrainingMetric
		err := rows.Scan(
			&m.ID, &m.Timestamp, &m.Epoch, &m.Step, &m.Loss,
			&m.LearningRate, &m.PaperID, &m.CheckpointID,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan metric row: %v", err)
		}
		metrics = append(metrics, m)
	}

	return metrics, nil
}

// calculateFileHash calculates SHA256 hash of a file
func calculateFileHash(filepath string) (string, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}

	return hex.EncodeToString(h.Sum(nil)), nil
}
