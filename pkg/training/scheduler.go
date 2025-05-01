package training

import "math"

// LRScheduler defines the interface for learning rate schedulers
type LRScheduler interface {
	GetLR(epoch, step int) float32
}

// CosineAnnealingLR implements cosine annealing learning rate schedule
type CosineAnnealingLR struct {
	initialLR float32
	minLR     float32
	tMax      int
}

// NewCosineAnnealingLR creates a new cosine annealing scheduler
func NewCosineAnnealingLR(initialLR, minLR float32, tMax int) *CosineAnnealingLR {
	return &CosineAnnealingLR{
		initialLR: initialLR,
		minLR:     minLR,
		tMax:      tMax,
	}
}

// GetLR returns the learning rate for the current step
func (s *CosineAnnealingLR) GetLR(epoch, step int) float32 {
	progress := float64(epoch*s.tMax+step) / float64(s.tMax)
	cosVal := 0.5 * (1.0 + math.Cos(math.Pi*progress))
	return s.minLR + float32(cosVal)*float32(s.initialLR-s.minLR)
}

// WarmupLR implements linear warmup followed by constant learning rate
type WarmupLR struct {
	initialLR   float32
	warmupSteps int
}

// NewWarmupLR creates a new warmup scheduler
func NewWarmupLR(initialLR float32, warmupSteps int) *WarmupLR {
	return &WarmupLR{
		initialLR:   initialLR,
		warmupSteps: warmupSteps,
	}
}

// GetLR returns the learning rate for the current step
func (s *WarmupLR) GetLR(epoch, step int) float32 {
	totalSteps := epoch*s.warmupSteps + step
	if totalSteps >= s.warmupSteps {
		return s.initialLR
	}
	return s.initialLR * float32(totalSteps) / float32(s.warmupSteps)
}

// LinearDecayLR implements linear learning rate decay
type LinearDecayLR struct {
	initialLR float32
	minLR     float32
	decayRate float32
}

// NewLinearDecayLR creates a new linear decay scheduler
func NewLinearDecayLR(initialLR, minLR, decayRate float32) *LinearDecayLR {
	return &LinearDecayLR{
		initialLR: initialLR,
		minLR:     minLR,
		decayRate: decayRate,
	}
}

// GetLR returns the learning rate for the current step
func (s *LinearDecayLR) GetLR(epoch, step int) float32 {
	lr := s.initialLR - float32(epoch)*s.decayRate
	if lr < s.minLR {
		return s.minLR
	}
	return lr
}
