package main

import "math"

type EQFilter struct {
	sampleRate float64
	frequency  float64
	gain       float64
	a1, a2     float64
	b0, b1, b2 float64
	x1, x2     float64
	y1, y2     float64
}

func NewEQFilter(sampleRate, frequency, gain float64) *EQFilter {
	// Convert frequency and gain to radians
	w0 := 2 * math.Pi * frequency / sampleRate
	alpha := math.Sin(w0) / (2 * math.Pow(10, gain/40))

	// Calculate filter coefficients
	b0 := (1 + math.Cos(w0)) / 2
	b1 := -(1 + math.Cos(w0))
	b2 := (1 + math.Cos(w0)) / 2
	a0 := 1 + alpha
	a1 := -2 * math.Cos(w0)
	a2 := 1 - alpha

	return &EQFilter{
		sampleRate: sampleRate,
		frequency:  frequency,
		gain:       gain,
		a1:         a1 / a0,
		a2:         a2 / a0,
		b0:         b0 / a0,
		b1:         b1 / a0,
		b2:         b2 / a0,
	}
}

func (f *EQFilter) Process(samples [][2]float64) {
	for i := range samples {
		x0 := samples[i][0]

		// Apply the equalizer filter
		y0 := f.b0*x0 + f.b1*f.x1 + f.b2*f.x2 - f.a1*f.y1 - f.a2*f.y2

		// Update filter state
		f.x2, f.x1 = f.x1, x0
		f.y2, f.y1 = f.y1, y0

		// Update the sample with the filtered value
		samples[i][0] = y0
		samples[i][1] = y0
	}
}

func (f *EQFilter) Err() error {
	return nil
}
