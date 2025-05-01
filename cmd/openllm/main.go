package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"openllm/pkg/database"
	"openllm/pkg/feeds"
	"openllm/pkg/model"
	"openllm/pkg/training"

	"github.com/gin-gonic/gin"
)

func main() {
	// Create necessary directories
	dirs := []string{"data", "checkpoints"}
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			log.Fatalf("Failed to create directory %s: %v", dir, err)
		}
	}

	// Initialize database
	db, err := database.NewDB("openllm.db")
	if err != nil {
		log.Fatalf("Failed to initialize database: %v", err)
	}
	defer db.Close()

	if err := db.Initialize(); err != nil {
		log.Fatalf("Failed to initialize database schema: %v", err)
	}

	// Create a channel for feeding content to the training system
	trainingChan := make(chan string, 100) // Buffer size of 100 articles

	// Initialize model
	config := &model.Config{
		VocabSize:    32000,
		MaxSeqLength: 512,
		HiddenSize:   768,
		NumLayers:    12,
		NumHeads:     12,
		FFNDim:       3072,
		Dropout:      0.1,
		LayerNormEps: 1e-5,
	}
	transformer := model.NewTransformer(config)

	// Initialize background learner with the training channel
	learner, err := training.NewBackgroundLearner(
		transformer,
		"data",        // Data directory
		"checkpoints", // Checkpoint directory
		32,            // Batch size
		0.001,         // Learning rate
		trainingChan,  // Channel for real-time content
	)
	if err != nil {
		log.Fatalf("Failed to create background learner: %v", err)
	}

	// Initialize feed manager
	feedManager := feeds.NewFeedManager(trainingChan)

	// Start background feed fetching (every 30 minutes)
	ctx := context.Background()
	feedManager.StartBackgroundFetching(ctx, 30*time.Minute)

	// Start background learning
	if err := learner.Start(ctx); err != nil {
		log.Fatalf("Failed to start background learner: %v", err)
	}
	defer learner.Stop()

	// Initialize Gin router
	router := gin.Default()

	// Basic health check endpoint
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status": "ok",
		})
	})

	// Training progress endpoint
	router.GET("/progress", func(c *gin.Context) {
		// Get recent metrics from database
		metrics, err := db.GetRecentMetrics(10)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		// Get learner progress
		progress := learner.GetProgress()
		progress["recent_metrics"] = metrics

		c.JSON(http.StatusOK, progress)
	})

	// List papers endpoint
	router.GET("/papers", func(c *gin.Context) {
		papers, err := db.GetPendingPapers()
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, papers)
	})

	// Add papers endpoint - accepts PDF/MD files for training
	router.POST("/papers", func(c *gin.Context) {
		file, err := c.FormFile("file")
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "No file provided"})
			return
		}

		// Check file extension
		ext := filepath.Ext(file.Filename)
		if ext != ".pdf" && ext != ".md" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Only PDF and MD files are supported"})
			return
		}

		// Save file to data directory
		dst := filepath.Join("data", file.Filename)
		if err := c.SaveUploadedFile(file, dst); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save file"})
			return
		}

		// Add paper to database
		paper, err := db.AddPaper(file.Filename, dst, ext[1:]) // ext[1:] removes the dot
		if err != nil {
			// Clean up file if database insert fails
			os.Remove(dst)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"message": "File uploaded successfully",
			"paper":   paper,
		})
	})

	// Start the server
	if err := router.Run(":8080"); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
