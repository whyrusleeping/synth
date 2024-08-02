package main

import (
	"fmt"
	"time"

	"github.com/gopxl/beep"
)

type Delay2 struct {
	buf       [][2]float64
	bufEffect Effect
	//buf2      [][2]float64
	delay    int
	decay    float64
	position int

	wpos int
}

func NewDelay2(amount time.Duration, decay float64) *Delay2 {
	sr := beep.SampleRate(sampleRate)
	n := sr.N(amount)

	delay := &Delay2{
		buf:   make([][2]float64, n*10),
		delay: n,
		decay: decay,
	}

	return delay
}

func (d *Delay2) GetSetter(k string) func(float64) {
	switch k {
	case "delay":
		return func(v float64) {
			d.delay = int(v)
			fmt.Println("delay is: ", d.delay)
		}
	case "decay":
		return func(v float64) {
			d.decay = v
		}

	default:
		return nil
	}

}

func (d *Delay2) ProcessSample(samples [][2]float64) {

	for i := range samples {
		samples[i][0] += d.buf[d.position%len(d.buf)][0]
		samples[i][1] += d.buf[d.position%len(d.buf)][1]

		dpos := (d.delay + d.position)
		wpos := d.wpos % len(d.buf)
		d.setBuffer(wpos, samples[i][0]*d.decay)
		if d.wpos < dpos {
			d.wpos++
			wpos = d.wpos % len(d.buf)
			d.setBuffer(wpos, samples[i][0]*d.decay)
		} else if d.wpos > dpos && dpos%3 == 0 {
			d.wpos--
		}

		d.position++
		d.wpos++
	}
}

func (d *Delay2) setBuffer(pos int, val float64) {
	if d.bufEffect != nil {
		var sample [1][2]float64
		sample[0][0] = val
		sample[0][1] = val
		d.bufEffect.ProcessSample(sample[:])
		val = sample[0][0]
	}

	d.buf[pos][0] = val
	d.buf[pos][1] = val
}

func (d *Delay2) Process(src beep.Streamer) beep.Streamer {
	return beep.StreamerFunc(func(samples [][2]float64) (n int, ok bool) {
		n, ok = src.Stream(samples)

		d.ProcessSample(samples)

		return n, ok
	})
}
