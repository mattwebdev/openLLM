package cli

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"openllm/pkg/metrics"
)

// CLI handles command-line interface for metrics
type CLI struct {
	tracker *metrics.MetricTracker
}

// NewCLI creates a new CLI instance
func NewCLI(tracker *metrics.MetricTracker) *CLI {
	return &CLI{
		tracker: tracker,
	}
}

// Run executes the CLI
func (c *CLI) Run(args []string) error {
	if len(args) < 2 {
		return fmt.Errorf("usage: %s <command> [options]", args[0])
	}

	cmd := args[1]
	switch cmd {
	case "show":
		return c.showMetrics(args[2:])
	case "export":
		return c.exportMetrics(args[2:])
	case "dashboard":
		return c.startDashboard(args[2:])
	default:
		return fmt.Errorf("unknown command: %s", cmd)
	}
}

// showMetrics displays metrics
func (c *CLI) showMetrics(args []string) error {
	fs := flag.NewFlagSet("show", flag.ExitOnError)
	epoch := fs.Int("epoch", -1, "Filter by epoch")
	fs.Parse(args)

	entries := c.tracker.GetMetrics()
	if *epoch >= 0 {
		var filtered []metrics.MetricEntry
		for _, entry := range entries {
			if entry.Epoch == *epoch {
				filtered = append(filtered, entry)
			}
		}
		entries = filtered
	}

	for _, entry := range entries {
		fmt.Printf("Epoch %d, Batch %d, Step %d:\n", entry.Epoch, entry.Batch, entry.Step)
		fmt.Printf("  Loss: %.4f, Learning Rate: %.6f\n", entry.Loss, entry.LearningRate)
		fmt.Printf("  Perplexity: %.4f, BLEU: %.4f, ROUGE: %.4f, Accuracy: %.4f\n",
			entry.Metrics.Perplexity, entry.Metrics.BLEU, entry.Metrics.ROUGE, entry.Metrics.Accuracy)
	}
	return nil
}

// exportMetrics exports metrics to a file
func (c *CLI) exportMetrics(args []string) error {
	fs := flag.NewFlagSet("export", flag.ExitOnError)
	output := fs.String("output", "metrics.json", "Output file path")
	fs.Parse(args)

	entries := c.tracker.GetMetrics()
	file, err := os.Create(*output)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(entries)
}

// startDashboard starts the metrics dashboard
func (c *CLI) startDashboard(args []string) error {
	fs := flag.NewFlagSet("dashboard", flag.ExitOnError)
	port := fs.String("port", ":8080", "Port to listen on")
	uiDir := fs.String("ui-dir", "frontend/dist", "Path to Svelte UI directory")
	fs.Parse(args)

	dashboard := metrics.NewDashboard(c.tracker, *uiDir)
	fmt.Printf("Starting dashboard on http://localhost%s\n", *port)
	return http.ListenAndServe(*port, dashboard)
}

// Helper functions
func parseTimeRange(s string) time.Duration {
	switch s {
	case "1h":
		return time.Hour
	case "6h":
		return 6 * time.Hour
	case "24h":
		return 24 * time.Hour
	default:
		return 0
	}
}

func (c *CLI) exportCSV(w io.Writer, metrics []metrics.MetricEntry) error {
	// Write header
	fmt.Fprintln(w, "timestamp,epoch,batch,step,loss,learning_rate,perplexity,bleu,rouge,accuracy")

	// Write data
	for _, m := range metrics {
		fmt.Fprintf(w, "%s,%d,%d,%d,%.4f,%.6f,%.2f,%.4f,%.4f,%.4f\n",
			m.Timestamp.Format(time.RFC3339),
			m.Epoch,
			m.Batch,
			m.Step,
			m.Loss,
			m.LearningRate,
			m.Metrics.Perplexity,
			m.Metrics.BLEU,
			m.Metrics.ROUGE,
			m.Metrics.Accuracy)
	}

	return nil
}
