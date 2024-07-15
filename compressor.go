package main

import (
	"fmt"
	"math"
)

type Compressor struct {
	threshold float64
	ratio     float64
	attack    float64
	release   float64
	makeup    float64
	envelope  float64
}

func NewCompressor(sampleRate, threshold, ratio, attack, release, makeup float64) *Compressor {
	return &Compressor{
		threshold: threshold,
		ratio:     ratio,
		attack:    math.Exp(-1 / (sampleRate * attack)),
		release:   math.Exp(-1 / (sampleRate * release)),
		makeup:    math.Pow(10, makeup/20),
		envelope:  0,
	}
}

func (c *Compressor) ProcessSample(samples [][2]float64) {
	for i := range samples {
		if samples[i][0] > 1 || samples[i][0] < -1 {
			fmt.Println("clipping")
		}
		// Get the maximum absolute value of left and right channels
		inputLevel := math.Max(math.Abs(samples[i][0]), math.Abs(samples[i][1]))

		// Convert input level to dB
		inputLevelDB := 20 * math.Log10(inputLevel)

		// Calculate the gain reduction
		gainReduction := 0.0
		if inputLevelDB > c.threshold {
			gainReduction = (inputLevelDB - c.threshold) * (1 - 1/c.ratio)
		}

		// Smooth the gain reduction (envelope follower)
		if gainReduction > c.envelope {
			c.envelope = c.attack*c.envelope + (1-c.attack)*gainReduction
		} else {
			c.envelope = c.release*c.envelope + (1-c.release)*gainReduction
		}

		// Apply the gain reduction and makeup gain
		gain := math.Pow(10, -c.envelope/20) * c.makeup

		// Apply the gain to both channels
		samples[i][0] *= gain
		samples[i][1] *= gain
	}
}

func (c *Compressor) GetSetter(k string) func(float64) {
	return nil
}
