package model

import (
	"errors"
	"math"

	"openllm/pkg/tensor"
)

var (
	ErrSequenceTooLong = errors.New("input sequence length exceeds maximum allowed length")
	ErrInvalidToken    = errors.New("invalid token ID")
)

// Config holds the configuration for the transformer model
type Config struct {
	VocabSize    int
	MaxSeqLength int
	HiddenSize   int
	NumLayers    int
	NumHeads     int
	FFNDim       int
	Dropout      float64
	LayerNormEps float64
}

// DefaultConfig returns a default configuration for the transformer
func DefaultConfig() *Config {
	return &Config{
		VocabSize:    32000,
		MaxSeqLength: 512,
		HiddenSize:   768,
		NumLayers:    12,
		NumHeads:     12,
		FFNDim:       3072,
		Dropout:      0.1,
		LayerNormEps: 1e-5,
	}
}

// Transformer represents the main transformer model
type Transformer struct {
	Config *Config

	// Embeddings
	WordEmbeddings     *tensor.Tensor
	PositionEmbeddings *tensor.Tensor

	// Transformer layers
	Layers []*TransformerLayer

	// Output layer
	OutputLayer *tensor.Tensor
}

// TransformerLayer represents a single transformer layer
type TransformerLayer struct {
	// Self-attention
	QueryWeights  *tensor.Tensor
	KeyWeights    *tensor.Tensor
	ValueWeights  *tensor.Tensor
	OutputWeights *tensor.Tensor

	// Feed-forward network
	FFNWeights1 *tensor.Tensor
	FFNWeights2 *tensor.Tensor

	// Layer normalization
	LayerNorm1 *tensor.Tensor
	LayerNorm2 *tensor.Tensor

	NumHeads     int
	MaxSeqLength int
}

// NewTransformer creates a new transformer model with the given configuration
func NewTransformer(config *Config) *Transformer {
	if config == nil {
		config = DefaultConfig()
	}

	// Initialize embeddings
	wordEmbeddings := tensor.NewTensor([]int{config.VocabSize, config.HiddenSize})
	positionEmbeddings := tensor.NewTensor([]int{config.MaxSeqLength, config.HiddenSize})

	// Initialize layers
	layers := make([]*TransformerLayer, config.NumLayers)
	for i := 0; i < config.NumLayers; i++ {
		layers[i] = &TransformerLayer{
			// Self-attention weights
			QueryWeights:  tensor.NewTensor([]int{config.HiddenSize, config.HiddenSize}),
			KeyWeights:    tensor.NewTensor([]int{config.HiddenSize, config.HiddenSize}),
			ValueWeights:  tensor.NewTensor([]int{config.HiddenSize, config.HiddenSize}),
			OutputWeights: tensor.NewTensor([]int{config.HiddenSize, config.HiddenSize}),

			// Feed-forward network weights
			FFNWeights1: tensor.NewTensor([]int{config.HiddenSize, config.FFNDim}),
			FFNWeights2: tensor.NewTensor([]int{config.FFNDim, config.HiddenSize}),

			// Layer normalization parameters
			LayerNorm1: tensor.NewTensor([]int{config.HiddenSize}),
			LayerNorm2: tensor.NewTensor([]int{config.HiddenSize}),
		}
	}

	// Initialize output layer
	outputLayer := tensor.NewTensor([]int{config.HiddenSize, config.VocabSize})

	return &Transformer{
		Config:             config,
		WordEmbeddings:     wordEmbeddings,
		PositionEmbeddings: positionEmbeddings,
		Layers:             layers,
		OutputLayer:        outputLayer,
	}
}

// Forward performs the forward pass through the transformer
func (t *Transformer) Forward(input []int) ([]float32, error) {
	// 1. Input embedding
	seqLen := len(input)
	if seqLen > t.Config.MaxSeqLength {
		return nil, ErrSequenceTooLong
	}

	// Get word embeddings
	embeddings := make([]float32, seqLen*t.Config.HiddenSize)
	for i, token := range input {
		if token < 0 || token >= t.Config.VocabSize {
			return nil, ErrInvalidToken
		}
		start := i * t.Config.HiddenSize
		copy(embeddings[start:start+t.Config.HiddenSize], t.WordEmbeddings.Data[token*t.Config.HiddenSize:])
	}

	// Add position embeddings
	for i := 0; i < seqLen; i++ {
		start := i * t.Config.HiddenSize
		for j := 0; j < t.Config.HiddenSize; j++ {
			embeddings[start+j] += t.PositionEmbeddings.Data[i*t.Config.HiddenSize+j]
		}
	}

	// 2. Process through transformer layers
	hidden := embeddings
	for _, layer := range t.Layers {
		var err error
		hidden, err = layer.Forward(hidden, seqLen, t.Config)
		if err != nil {
			return nil, err
		}
	}

	// 3. Output layer
	output := make([]float32, seqLen*t.Config.VocabSize)
	for i := 0; i < seqLen; i++ {
		start := i * t.Config.HiddenSize
		hiddenSlice := hidden[start : start+t.Config.HiddenSize]
		outputSlice := output[i*t.Config.VocabSize : (i+1)*t.Config.VocabSize]

		// Matrix multiplication with output layer
		for j := 0; j < t.Config.VocabSize; j++ {
			sum := float32(0)
			for k := 0; k < t.Config.HiddenSize; k++ {
				sum += hiddenSlice[k] * t.OutputLayer.Data[k*t.Config.VocabSize+j]
			}
			outputSlice[j] = sum
		}
	}

	return output, nil
}

// Forward performs the forward pass through a single transformer layer
func (l *TransformerLayer) Forward(input []float32, seqLen int, config *Config) ([]float32, error) {
	// 1. Self-attention
	// Query, Key, Value projections
	query := make([]float32, seqLen*config.HiddenSize)
	key := make([]float32, seqLen*config.HiddenSize)
	value := make([]float32, seqLen*config.HiddenSize)

	for i := 0; i < seqLen; i++ {
		start := i * config.HiddenSize
		inputSlice := input[start : start+config.HiddenSize]

		// Project to query, key, value
		for j := 0; j < config.HiddenSize; j++ {
			qSum, kSum, vSum := float32(0), float32(0), float32(0)
			for k := 0; k < config.HiddenSize; k++ {
				qSum += inputSlice[k] * l.QueryWeights.Data[k*config.HiddenSize+j]
				kSum += inputSlice[k] * l.KeyWeights.Data[k*config.HiddenSize+j]
				vSum += inputSlice[k] * l.ValueWeights.Data[k*config.HiddenSize+j]
			}
			query[start+j] = qSum
			key[start+j] = kSum
			value[start+j] = vSum
		}
	}

	// 2. Scaled dot-product attention
	attention := make([]float32, seqLen*seqLen)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			sum := float32(0)
			for k := 0; k < config.HiddenSize; k++ {
				sum += query[i*config.HiddenSize+k] * key[j*config.HiddenSize+k]
			}
			attention[i*seqLen+j] = sum / float32(math.Sqrt(float64(config.HiddenSize)))
		}
	}

	// Apply softmax
	attentionTensor, err := tensor.NewTensorFromData([]int{seqLen, seqLen}, attention)
	if err != nil {
		return nil, err
	}
	attention = attentionTensor.Softmax().Data

	// 3. Apply attention to values
	attended := make([]float32, seqLen*config.HiddenSize)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < config.HiddenSize; j++ {
			sum := float32(0)
			for k := 0; k < seqLen; k++ {
				sum += attention[i*seqLen+k] * value[k*config.HiddenSize+j]
			}
			attended[i*config.HiddenSize+j] = sum
		}
	}

	// 4. Output projection
	output := make([]float32, seqLen*config.HiddenSize)
	for i := 0; i < seqLen; i++ {
		start := i * config.HiddenSize
		attendedSlice := attended[start : start+config.HiddenSize]
		outputSlice := output[start : start+config.HiddenSize]

		for j := 0; j < config.HiddenSize; j++ {
			sum := float32(0)
			for k := 0; k < config.HiddenSize; k++ {
				sum += attendedSlice[k] * l.OutputWeights.Data[k*config.HiddenSize+j]
			}
			outputSlice[j] = sum
		}
	}

	// 5. Add & Norm
	for i := 0; i < seqLen; i++ {
		start := i * config.HiddenSize
		outputSlice := output[start : start+config.HiddenSize]
		inputSlice := input[start : start+config.HiddenSize]

		// Add residual connection
		for j := 0; j < config.HiddenSize; j++ {
			outputSlice[j] += inputSlice[j]
		}

		// Layer normalization
		normalized := LayerNorm(outputSlice, config.LayerNormEps)
		copy(outputSlice, normalized)
	}

	// 6. Feed-forward network
	ffnOutput := make([]float32, seqLen*config.HiddenSize)
	for i := 0; i < seqLen; i++ {
		start := i * config.HiddenSize
		outputSlice := output[start : start+config.HiddenSize]
		ffnSlice := ffnOutput[start : start+config.HiddenSize]

		// First linear layer
		hidden := make([]float32, config.FFNDim)
		for j := 0; j < config.FFNDim; j++ {
			sum := float32(0)
			for k := 0; k < config.HiddenSize; k++ {
				sum += outputSlice[k] * l.FFNWeights1.Data[k*config.FFNDim+j]
			}
			hidden[j] = sum
		}

		// GELU activation
		for j := 0; j < config.FFNDim; j++ {
			hidden[j] = float32(0.5) * hidden[j] * (1 + float32(math.Tanh(float64(math.Sqrt(2/math.Pi)*(float64(hidden[j])+0.044715*math.Pow(float64(hidden[j]), 3))))))
		}

		// Second linear layer
		for j := 0; j < config.HiddenSize; j++ {
			sum := float32(0)
			for k := 0; k < config.FFNDim; k++ {
				sum += hidden[k] * l.FFNWeights2.Data[k*config.HiddenSize+j]
			}
			ffnSlice[j] = sum
		}
	}

	// 7. Final Add & Norm
	for i := 0; i < seqLen; i++ {
		start := i * config.HiddenSize
		ffnSlice := ffnOutput[start : start+config.HiddenSize]
		outputSlice := output[start : start+config.HiddenSize]

		// Add residual connection
		for j := 0; j < config.HiddenSize; j++ {
			ffnSlice[j] += outputSlice[j]
		}

		// Layer normalization
		normalized := LayerNorm(ffnSlice, config.LayerNormEps)
		copy(ffnSlice, normalized)
	}

	return ffnOutput, nil
}

// LayerNorm implements layer normalization
func LayerNorm(x []float32, eps float64) []float32 {
	mean := 0.0
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(len(x))

	variance := 0.0
	for _, v := range x {
		diff := float64(v) - mean
		variance += diff * diff
	}
	variance /= float64(len(x))

	result := make([]float32, len(x))
	for i, v := range x {
		result[i] = float32((float64(v) - mean) / math.Sqrt(variance+eps))
	}
	return result
}

// Clone creates a deep copy of the transformer model
func (t *Transformer) Clone() *Transformer {
	// Clone config
	config := &Config{
		VocabSize:    t.Config.VocabSize,
		MaxSeqLength: t.Config.MaxSeqLength,
		HiddenSize:   t.Config.HiddenSize,
		NumLayers:    t.Config.NumLayers,
		NumHeads:     t.Config.NumHeads,
		FFNDim:       t.Config.FFNDim,
		Dropout:      t.Config.Dropout,
		LayerNormEps: t.Config.LayerNormEps,
	}

	// Clone embeddings
	wordEmbeddings := tensor.NewTensor(t.WordEmbeddings.Shape)
	copy(wordEmbeddings.Data, t.WordEmbeddings.Data)

	positionEmbeddings := tensor.NewTensor(t.PositionEmbeddings.Shape)
	copy(positionEmbeddings.Data, t.PositionEmbeddings.Data)

	// Clone layers
	layers := make([]*TransformerLayer, len(t.Layers))
	for i, layer := range t.Layers {
		layers[i] = &TransformerLayer{
			// Clone self-attention weights
			QueryWeights:  tensor.NewTensor(layer.QueryWeights.Shape),
			KeyWeights:    tensor.NewTensor(layer.KeyWeights.Shape),
			ValueWeights:  tensor.NewTensor(layer.ValueWeights.Shape),
			OutputWeights: tensor.NewTensor(layer.OutputWeights.Shape),

			// Clone feed-forward network weights
			FFNWeights1: tensor.NewTensor(layer.FFNWeights1.Shape),
			FFNWeights2: tensor.NewTensor(layer.FFNWeights2.Shape),

			// Clone layer normalization parameters
			LayerNorm1: tensor.NewTensor(layer.LayerNorm1.Shape),
			LayerNorm2: tensor.NewTensor(layer.LayerNorm2.Shape),

			NumHeads:     layer.NumHeads,
			MaxSeqLength: layer.MaxSeqLength,
		}

		// Copy data
		copy(layers[i].QueryWeights.Data, layer.QueryWeights.Data)
		copy(layers[i].KeyWeights.Data, layer.KeyWeights.Data)
		copy(layers[i].ValueWeights.Data, layer.ValueWeights.Data)
		copy(layers[i].OutputWeights.Data, layer.OutputWeights.Data)
		copy(layers[i].FFNWeights1.Data, layer.FFNWeights1.Data)
		copy(layers[i].FFNWeights2.Data, layer.FFNWeights2.Data)
		copy(layers[i].LayerNorm1.Data, layer.LayerNorm1.Data)
		copy(layers[i].LayerNorm2.Data, layer.LayerNorm2.Data)
	}

	// Clone output layer
	outputLayer := tensor.NewTensor(t.OutputLayer.Shape)
	copy(outputLayer.Data, t.OutputLayer.Data)

	return &Transformer{
		Config:             config,
		WordEmbeddings:     wordEmbeddings,
		PositionEmbeddings: positionEmbeddings,
		Layers:             layers,
		OutputLayer:        outputLayer,
	}
}
