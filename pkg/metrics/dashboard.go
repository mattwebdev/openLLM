package metrics

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
)

// Dashboard handles the metrics visualization interface
type Dashboard struct {
	tracker *MetricTracker
	uiDir   string
}

// NewDashboard creates a new metrics dashboard
func NewDashboard(tracker *MetricTracker, uiDir string) *Dashboard {
	return &Dashboard{
		tracker: tracker,
		uiDir:   uiDir,
	}
}

// ServeHTTP handles HTTP requests for the dashboard
func (d *Dashboard) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/":
		d.serveFrontend(w, r)
	case "/api/metrics":
		d.serveMetrics(w, r)
	case "/api/metrics/summary":
		d.serveMetricsSummary(w, r)
	default:
		// Serve static files from the UI directory
		http.ServeFile(w, r, filepath.Join(d.uiDir, r.URL.Path[1:]))
	}
}

// serveFrontend serves the Svelte frontend
func (d *Dashboard) serveFrontend(w http.ResponseWriter, r *http.Request) {
	indexPath := filepath.Join(d.uiDir, "index.html")
	if _, err := os.Stat(indexPath); os.IsNotExist(err) {
		http.Error(w, "Frontend not found. Please build the Svelte app first.", http.StatusNotFound)
		return
	}
	http.ServeFile(w, r, indexPath)
}

// serveMetrics serves the metrics data as JSON
func (d *Dashboard) serveMetrics(w http.ResponseWriter, r *http.Request) {
	metrics := d.tracker.GetMetrics()
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(metrics); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

// serveMetricsSummary serves a summary of metrics as JSON
func (d *Dashboard) serveMetricsSummary(w http.ResponseWriter, r *http.Request) {
	epoch := 0 // Default to first epoch
	if e := r.URL.Query().Get("epoch"); e != "" {
		fmt.Sscanf(e, "%d", &epoch)
	}

	summary := d.tracker.GetMetricsSummary(epoch)
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(summary); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}
