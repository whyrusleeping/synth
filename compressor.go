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
	envelope  float64

	rollingAv  float64
	avInterval float64
}

func NewCompressor(threshold, ratio, attack, release float64) *Compressor {
	return &Compressor{
		threshold: threshold,
		ratio:     ratio,
		attack:    attack,  //math.Exp(-1 / (sampleRate * attack)),
		release:   release, //math.Exp(-1 / (sampleRate * release)),
		envelope:  0,

		avInterval: 500,
	}
}

func (c *Compressor) compressValue(v float64) float64 {
	// some really basic recentering, probably want to just to min-max
	// computation and center between those values instead
	c.rollingAv = ((c.rollingAv * (c.avInterval - 1)) + v) / c.avInterval
	v -= c.rollingAv

	if math.Abs(v) > c.threshold {
		c.envelope += (math.Abs(v) - c.envelope) * c.attack
	} else {
		c.envelope += (math.Abs(v) - c.envelope) * c.release
	}

	if c.envelope <= c.threshold {
		return v
	}

	gain_reduction := math.Pow((c.threshold / c.envelope), c.ratio)

	return v * gain_reduction
}

func (c *Compressor) ProcessSample(samples [][2]float64) {
	for i := range samples {
		if samples[i][0] > 1 || samples[i][0] < -1 {
			//fmt.Println("clipping")
		}

		val := c.compressValue(samples[i][0])
		if math.Abs(val) > 1 {
			fmt.Println("clipping")
		}

		// Apply the gain to both channels
		samples[i][0] = val
		samples[i][1] = val
	}
}

func (c *Compressor) GetSetter(k string) func(float64) {
	return nil
}
