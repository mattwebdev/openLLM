package optimization

import (
	"fmt"
	"math"
	"strings"
	"sync"

	"openllm/pkg/model"
)

// CompressionConfig holds configuration for model compression
type CompressionConfig struct {
	Method           string  // "svd", "tucker", "cp", or "hybrid"
	CompressionRatio float64 // Target compression ratio (0.0 to 1.0)
	RankRatio        float64 // Ratio of original rank to keep
	ErrorThreshold   float64 // Maximum allowed error
	EnableMixedComp  bool    // Enable mixed compression strategies
	EnableProgComp   bool    // Enable progressive compression
}

// Compressor handles model compression
type Compressor struct {
	config CompressionConfig
	stats  map[string]interface{}
	mu     sync.RWMutex
}

// NewCompressor creates a new compressor instance
func NewCompressor(config CompressionConfig) *Compressor {
	return &Compressor{
		config: config,
		stats:  make(map[string]interface{}),
	}
}

// CompressModel applies compression to the model
func (c *Compressor) CompressModel(model *model.Transformer) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	switch c.config.Method {
	case "svd":
		return c.applySVDCompression(model)
	case "tucker":
		return c.applyTuckerCompression(model)
	case "cp":
		return c.applyCPCompression(model)
	case "hybrid":
		return c.applyHybridCompression(model)
	default:
		return fmt.Errorf("unsupported compression method: %s", c.config.Method)
	}
}

// applySVDCompression applies SVD-based compression
func (c *Compressor) applySVDCompression(model *model.Transformer) error {
	for i, layer := range model.Layers {
		// Compress attention weights
		if err := c.compressWithSVD(layer.QueryWeights.Data, fmt.Sprintf("layer_%d/query", i)); err != nil {
			return err
		}
		if err := c.compressWithSVD(layer.KeyWeights.Data, fmt.Sprintf("layer_%d/key", i)); err != nil {
			return err
		}
		if err := c.compressWithSVD(layer.ValueWeights.Data, fmt.Sprintf("layer_%d/value", i)); err != nil {
			return err
		}

		// Compress FFN weights
		if err := c.compressWithSVD(layer.FFNWeights1.Data, fmt.Sprintf("layer_%d/ffn1", i)); err != nil {
			return err
		}
		if err := c.compressWithSVD(layer.FFNWeights2.Data, fmt.Sprintf("layer_%d/ffn2", i)); err != nil {
			return err
		}
	}

	return nil
}

// compressWithSVD applies SVD compression to a weight matrix
func (c *Compressor) compressWithSVD(weights []float32, name string) error {
	if len(weights) == 0 {
		return nil
	}

	// Convert to 2D matrix
	rows := int(math.Sqrt(float64(len(weights))))
	if rows*rows != len(weights) {
		return fmt.Errorf("weights length must be perfect square, got %d", len(weights))
	}

	// Perform SVD
	u, s, v := c.computeSVD(weights, rows)

	// Determine rank based on compression ratio
	targetRank := int(float64(rows) * c.config.RankRatio)
	if targetRank < 1 {
		targetRank = 1
	}

	// Truncate matrices
	u = u[:rows*targetRank]
	s = s[:targetRank]
	v = v[:targetRank*rows]

	// Reconstruct compressed matrix
	compressed := make([]float32, rows*rows)
	c.reconstructFromSVD(compressed, u, s, v, rows, targetRank)

	// Calculate compression error
	error := c.calculateError(weights, compressed)
	if error > c.config.ErrorThreshold {
		return fmt.Errorf("compression error %f exceeds threshold %f", error, c.config.ErrorThreshold)
	}

	// Update weights
	copy(weights, compressed)

	// Update stats
	c.stats[name+"_rank"] = targetRank
	c.stats[name+"_error"] = error
	c.stats[name+"_compression_ratio"] = float64(len(compressed)) / float64(len(weights))

	return nil
}

// applyTuckerCompression applies Tucker decomposition
func (c *Compressor) applyTuckerCompression(model *model.Transformer) error {
	for i, layer := range model.Layers {
		// Compress attention weights using Tucker decomposition
		if err := c.compressWithTucker(layer.QueryWeights.Data, fmt.Sprintf("layer_%d/query", i)); err != nil {
			return err
		}
		if err := c.compressWithTucker(layer.KeyWeights.Data, fmt.Sprintf("layer_%d/key", i)); err != nil {
			return err
		}
		if err := c.compressWithTucker(layer.ValueWeights.Data, fmt.Sprintf("layer_%d/value", i)); err != nil {
			return err
		}
	}

	return nil
}

// compressWithTucker applies Tucker decomposition
func (c *Compressor) compressWithTucker(weights []float32, name string) error {
	if len(weights) == 0 {
		return nil
	}

	// Convert to 3D tensor
	dim := int(math.Cbrt(float64(len(weights))))
	if dim*dim*dim != len(weights) {
		return fmt.Errorf("weights length must be perfect cube, got %d", len(weights))
	}

	// Compute Tucker decomposition
	core, factors := c.computeTucker(weights, dim)

	// Determine rank based on compression ratio
	targetRank := int(float64(dim) * c.config.RankRatio)
	if targetRank < 1 {
		targetRank = 1
	}

	// Truncate factors
	for i := range factors {
		factors[i] = factors[i][:targetRank*dim]
	}

	// Reconstruct compressed tensor
	compressed := make([]float32, dim*dim*dim)
	c.reconstructFromTucker(compressed, core, factors, dim, targetRank)

	// Calculate compression error
	error := c.calculateError(weights, compressed)
	if error > c.config.ErrorThreshold {
		return fmt.Errorf("compression error %f exceeds threshold %f", error, c.config.ErrorThreshold)
	}

	// Update weights
	copy(weights, compressed)

	// Update stats
	c.stats[name+"_rank"] = targetRank
	c.stats[name+"_error"] = error
	c.stats[name+"_compression_ratio"] = float64(len(compressed)) / float64(len(weights))

	return nil
}

// applyCPCompression applies CP decomposition
func (c *Compressor) applyCPCompression(model *model.Transformer) error {
	for i, layer := range model.Layers {
		// Compress FFN weights using CP decomposition
		if err := c.compressWithCP(layer.FFNWeights1.Data, fmt.Sprintf("layer_%d/ffn1", i)); err != nil {
			return err
		}
		if err := c.compressWithCP(layer.FFNWeights2.Data, fmt.Sprintf("layer_%d/ffn2", i)); err != nil {
			return err
		}
	}

	return nil
}

// compressWithCP applies CP decomposition
func (c *Compressor) compressWithCP(weights []float32, name string) error {
	if len(weights) == 0 {
		return nil
	}

	// Convert to 3D tensor
	dim := int(math.Cbrt(float64(len(weights))))
	if dim*dim*dim != len(weights) {
		return fmt.Errorf("weights length must be perfect cube, got %d", len(weights))
	}

	// Compute CP decomposition
	factors := c.computeCP(weights, dim)

	// Determine rank based on compression ratio
	targetRank := int(float64(dim) * c.config.RankRatio)
	if targetRank < 1 {
		targetRank = 1
	}

	// Truncate factors
	for i := range factors {
		factors[i] = factors[i][:targetRank*dim]
	}

	// Reconstruct compressed tensor
	compressed := make([]float32, dim*dim*dim)
	c.reconstructFromCP(compressed, factors, dim, targetRank)

	// Calculate compression error
	error := c.calculateError(weights, compressed)
	if error > c.config.ErrorThreshold {
		return fmt.Errorf("compression error %f exceeds threshold %f", error, c.config.ErrorThreshold)
	}

	// Update weights
	copy(weights, compressed)

	// Update stats
	c.stats[name+"_rank"] = targetRank
	c.stats[name+"_error"] = error
	c.stats[name+"_compression_ratio"] = float64(len(compressed)) / float64(len(weights))

	return nil
}

// applyHybridCompression applies mixed compression strategies
func (c *Compressor) applyHybridCompression(model *model.Transformer) error {
	for i, layer := range model.Layers {
		// Use SVD for attention weights
		if err := c.compressWithSVD(layer.QueryWeights.Data, fmt.Sprintf("layer_%d/query", i)); err != nil {
			return err
		}
		if err := c.compressWithSVD(layer.KeyWeights.Data, fmt.Sprintf("layer_%d/key", i)); err != nil {
			return err
		}
		if err := c.compressWithSVD(layer.ValueWeights.Data, fmt.Sprintf("layer_%d/value", i)); err != nil {
			return err
		}

		// Use Tucker for FFN weights
		if err := c.compressWithTucker(layer.FFNWeights1.Data, fmt.Sprintf("layer_%d/ffn1", i)); err != nil {
			return err
		}
		if err := c.compressWithCP(layer.FFNWeights2.Data, fmt.Sprintf("layer_%d/ffn2", i)); err != nil {
			return err
		}
	}

	return nil
}

// Helper functions for matrix/tensor operations

func (c *Compressor) computeSVD(matrix []float32, dim int) ([]float32, []float32, []float32) {
	// Placeholder: In a real implementation, use a proper linear algebra library
	u := make([]float32, dim*dim)
	s := make([]float32, dim)
	v := make([]float32, dim*dim)
	return u, s, v
}

func (c *Compressor) computeTucker(tensor []float32, dim int) ([]float32, [][]float32) {
	// Placeholder: In a real implementation, use a proper tensor algebra library
	core := make([]float32, dim*dim*dim)
	factors := make([][]float32, 3)
	for i := range factors {
		factors[i] = make([]float32, dim*dim)
	}
	return core, factors
}

func (c *Compressor) computeCP(tensor []float32, dim int) [][]float32 {
	// Placeholder: In a real implementation, use a proper tensor algebra library
	factors := make([][]float32, 3)
	for i := range factors {
		factors[i] = make([]float32, dim*dim)
	}
	return factors
}

func (c *Compressor) reconstructFromSVD(result, u []float32, s []float32, v []float32, dim, rank int) {
	// Placeholder: In a real implementation, use proper matrix multiplication
	for i := 0; i < dim*dim; i++ {
		result[i] = 0
		for j := 0; j < rank; j++ {
			result[i] += u[i*rank+j] * s[j] * v[j*dim+(i%dim)]
		}
	}
}

func (c *Compressor) reconstructFromTucker(result, core []float32, factors [][]float32, dim, rank int) {
	// Placeholder: In a real implementation, use proper tensor operations
	for i := 0; i < dim*dim*dim; i++ {
		result[i] = core[i]
	}
}

func (c *Compressor) reconstructFromCP(result []float32, factors [][]float32, dim, rank int) {
	// Placeholder: In a real implementation, use proper tensor operations
	for i := 0; i < dim*dim*dim; i++ {
		result[i] = 0
		for r := 0; r < rank; r++ {
			prod := float32(1.0)
			for j := 0; j < 3; j++ {
				prod *= factors[j][r*dim+(i>>(uint(j)*8))%dim]
			}
			result[i] += prod
		}
	}
}

func (c *Compressor) calculateError(original, compressed []float32) float64 {
	if len(original) != len(compressed) {
		return math.MaxFloat64
	}

	var sumSquaredError float64
	var sumSquaredOriginal float64

	for i := range original {
		diff := float64(original[i] - compressed[i])
		sumSquaredError += diff * diff
		sumSquaredOriginal += float64(original[i] * original[i])
	}

	if sumSquaredOriginal == 0 {
		return 0
	}

	return math.Sqrt(sumSquaredError / sumSquaredOriginal)
}

// GetCompressionStats returns statistics about the compression
func (c *Compressor) GetCompressionStats() map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()

	stats := make(map[string]interface{})
	for k, v := range c.stats {
		stats[k] = v
	}

	// Calculate average compression ratio and error
	var totalRatio, totalError float64
	var count int

	for name, value := range c.stats {
		if strings.HasSuffix(name, "_compression_ratio") {
			totalRatio += value.(float64)
			count++
		} else if strings.HasSuffix(name, "_error") {
			totalError += value.(float64)
		}
	}

	if count > 0 {
		stats["average_compression_ratio"] = totalRatio / float64(count)
		stats["average_error"] = totalError / float64(count)
	}

	stats["method"] = c.config.Method
	stats["target_ratio"] = c.config.CompressionRatio
	stats["rank_ratio"] = c.config.RankRatio
	stats["error_threshold"] = c.config.ErrorThreshold

	return stats
}
