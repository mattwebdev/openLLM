package metrics

import (
	"encoding/json"
	"os"
	"sync"
	"time"
)

// Metrics represents the evaluation metrics
type Metrics struct {
	Perplexity float64
	BLEU       float64
	ROUGE      float64
	Accuracy   float64
}

// MetricEntry represents a single metric entry with metadata
type MetricEntry struct {
	Timestamp    time.Time
	Epoch        int
	Batch        int
	Step         int
	Loss         float64
	LearningRate float64
	Metrics      Metrics
}

// MetricTracker handles logging and tracking of metrics
type MetricTracker struct {
	mu       sync.Mutex
	entries  []MetricEntry
	file     *os.File
	flushDur time.Duration
}

// NewMetricTracker creates a new metric tracker
func NewMetricTracker(logFile string, flushInterval time.Duration) (*MetricTracker, error) {
	file, err := os.OpenFile(logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return nil, err
	}

	tracker := &MetricTracker{
		file:     file,
		flushDur: flushInterval,
	}

	// Start periodic flushing
	go tracker.periodicFlush()

	return tracker, nil
}

// LogMetrics logs a new set of metrics
func (t *MetricTracker) LogMetrics(metrics Metrics, epoch, batch, step int, loss, learningRate float64) error {
	entry := MetricEntry{
		Timestamp:    time.Now(),
		Epoch:        epoch,
		Batch:        batch,
		Step:         step,
		Loss:         loss,
		LearningRate: learningRate,
		Metrics:      metrics,
	}

	t.mu.Lock()
	t.entries = append(t.entries, entry)
	t.mu.Unlock()

	return t.flush()
}

// GetMetrics returns all logged metrics
func (t *MetricTracker) GetMetrics() []MetricEntry {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.entries
}

// GetMetricsSummary returns a summary of metrics for a specific epoch
func (t *MetricTracker) GetMetricsSummary(epoch int) Metrics {
	t.mu.Lock()
	defer t.mu.Unlock()

	var sum Metrics
	var count int

	for _, entry := range t.entries {
		if entry.Epoch == epoch {
			sum.Perplexity += entry.Metrics.Perplexity
			sum.BLEU += entry.Metrics.BLEU
			sum.ROUGE += entry.Metrics.ROUGE
			sum.Accuracy += entry.Metrics.Accuracy
			count++
		}
	}

	if count > 0 {
		sum.Perplexity /= float64(count)
		sum.BLEU /= float64(count)
		sum.ROUGE /= float64(count)
		sum.Accuracy /= float64(count)
	}

	return sum
}

// Close closes the metric tracker and its associated file
func (t *MetricTracker) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if err := t.flush(); err != nil {
		return err
	}
	return t.file.Close()
}

// flush writes the current entries to the log file
func (t *MetricTracker) flush() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if len(t.entries) == 0 {
		return nil
	}

	encoder := json.NewEncoder(t.file)
	for _, entry := range t.entries {
		if err := encoder.Encode(entry); err != nil {
			return err
		}
	}

	t.entries = nil
	return nil
}

// periodicFlush periodically flushes the metrics to disk
func (t *MetricTracker) periodicFlush() {
	ticker := time.NewTicker(t.flushDur)
	defer ticker.Stop()

	for range ticker.C {
		if err := t.flush(); err != nil {
			// Log error but continue
			continue
		}
	}
}
