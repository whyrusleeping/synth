package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime/pprof"
	"strings"
	"sync"
	"time"

	"github.com/gopxl/beep"
	"github.com/gopxl/beep/speaker"
	"github.com/rakyll/portmidi"
)

type OscFunc func(float64) float64

const (
	sampleRate = 44100
	duration   = 2
)

var sr = beep.SampleRate(sampleRate)

type Effect interface {
	ProcessSample(samples [][2]float64)
	GetSetter(k string) func(float64)
}

type Recorder struct {
	lk       sync.Mutex
	buf      [][2]float64
	position int

	sub beep.Streamer
}

type EffectsStack struct {
	names   []string
	effects []Effect
}

func (es *EffectsStack) ProcessSample(samples [][2]float64) {
	for _, e := range es.effects {
		e.ProcessSample(samples)
	}
}

func (es *EffectsStack) AddEffect(name string, effect Effect) {
	es.names = append(es.names, name)
	es.effects = append(es.effects, effect)
}

func (es *EffectsStack) GetSetter(k string) func(float64) {
	parts := strings.Split(k, ".")
	sub := strings.Join(parts[1:], ".")
	for i, n := range es.names {
		if parts[0] == n {
			return es.effects[i].GetSetter(sub)
		}
	}
	return nil
}

func (r *Recorder) Stream(samples [][2]float64) (int, bool) {
	n, ok := r.sub.Stream(samples)
	if !ok {
		return n, ok
	}

	r.lk.Lock()
	defer r.lk.Unlock()

	for i := range samples {
		ix := r.position % len(r.buf)
		r.buf[ix][0] = samples[i][0]
		r.buf[ix][1] = samples[i][1]
		r.position++
	}
	return n, ok
}

func (r *Recorder) GetSnapshot(buf [][2]float64) int {
	r.lk.Lock()
	defer r.lk.Unlock()

	lim := len(buf)
	if len(r.buf) < lim {
		lim = len(r.buf)
	}

	for i := 0; i < lim; i++ {
		ix := (r.position + i) % len(r.buf)
		buf[i] = r.buf[ix]
	}

	return lim
}

func (r *Recorder) Err() error {
	return nil
}

func MulMix(a, b beep.Streamer) beep.StreamerFunc {
	abuf := make([][2]float64, 512)
	bbuf := make([][2]float64, 512)
	return func(samples [][2]float64) (int, bool) {
		olen := len(samples)
		for len(samples) > 0 {
			n := len(abuf)
			if n > len(samples) {
				n = len(samples)
			}
			an, aok := a.Stream(abuf[:n])
			bn, bok := b.Stream(bbuf[:n])

			if an != bn {
				panic("not dealing with this yet")
			}
			if !aok || !bok {
				panic("not dealing with this yet either")
			}

			for i := 0; i < n; i++ {
				samples[i][0] = abuf[i][0] * bbuf[i][0]
				samples[i][1] = abuf[i][1] * bbuf[i][1]
			}

			samples = samples[n:]
		}
		return olen, true
	}
}

type Delay struct {
	buf      [][2]float64
	delay    int
	decay    float64
	position int

	wpos int
}

func (d *Delay) GetSetter(k string) func(float64) {
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

func (d *Delay) ProcessSample(samples [][2]float64) {

	/*
		if d.delay != d.lastDelay {

		}

		odpos := (d.lastPosition + 1) % len(d.buf)
		ndpos := (d.delay + d.position) % len(d.buf)
		if odpos != ndpos {
			// some sort of adjustment was made to the delay
			// should probably do something fancy, but for now just repeat the last delay buffer value until we 'catch up'
		}
	*/

	for i := range samples {
		samples[i][0] += d.buf[d.position%len(d.buf)][0]
		samples[i][1] += d.buf[d.position%len(d.buf)][1]

		dpos := (d.delay + d.position)
		wpos := d.wpos % len(d.buf)
		d.buf[wpos][0] = samples[i][0] * d.decay
		d.buf[wpos][1] = samples[i][1] * d.decay
		if d.wpos < dpos {
			d.wpos++
			wpos = d.wpos % len(d.buf)
			d.buf[wpos][0] = samples[i][0] * d.decay
			d.buf[wpos][1] = samples[i][1] * d.decay
		} else if d.wpos > dpos && dpos%3 == 0 {
			d.wpos--
		}

		d.position++
		d.wpos++
	}
}

func (d *Delay) Process(src beep.Streamer) beep.Streamer {
	return beep.StreamerFunc(func(samples [][2]float64) (n int, ok bool) {
		n, ok = src.Stream(samples)

		d.ProcessSample(samples)

		return n, ok
	})
}

type Sweeper struct {
	Action func(float64)
	val    float64
	dir    int
}

func (s *Sweeper) Run() {
	for range time.Tick(time.Millisecond * 5) {
		if s.dir == 0 {
			s.val = 0
			s.dir = 1
		}

		if s.dir > 0 {
			s.val += 0.01
		} else {
			s.val -= 0.01
		}

		if s.val >= 1 {
			s.val = 1
			s.dir = -1
		}
		if s.val <= 0 {
			s.val = 0
			s.dir = 1
		}

		s.Action(s.val)
	}
}

// butterworth filter
type Butterworth struct {
	b0, b1, b2, a1, a2 float64
	x1, x2, y1, y2     float64
	sampleRate         float64

	position int

	mode int

	frequency float64
	width     float64

	frequencyEnv *PureEnv
	widthEnv     *PureEnv
}

const (
	ModeLowPass  = 1
	ModeBandPass = 2
)

func NewLowPass(cutoffFreq, sampleRate float64) *Butterworth {
	b := &Butterworth{
		sampleRate: sampleRate,
	}

	b.UpdateCutoff(cutoffFreq)
	return b
}

func (b *Butterworth) UpdateCutoff(cutoffFreq float64) {
	b.mode = ModeLowPass
	b.frequency = cutoffFreq

	// Convert cutoff frequency to radians
	wc := 2 * math.Pi * cutoffFreq / sampleRate

	// Compute filter coefficients
	cosw := math.Cos(wc)
	alpha := math.Sin(wc) / (2 * 0.707) // Q = 0.707 for Butterworth filter

	b0 := (1 - cosw) / 2
	b1 := 1 - cosw
	b2 := (1 - cosw) / 2
	a0 := 1 + alpha
	a1 := -2 * cosw
	a2 := 1 - alpha

	// Normalize filter coefficients
	b0 /= a0
	b1 /= a0
	b2 /= a0
	a1 /= a0
	a2 /= a0

	b.b0 = b0
	b.b1 = b1
	b.b2 = b2
	b.a1 = a1
	b.a2 = a2
}

func NewBandPass(sampleRate, frequency, width float64) *Butterworth {
	b := &Butterworth{
		sampleRate: sampleRate,
	}

	b.UpdateBandpass(frequency, width)
	return b
}

func (b *Butterworth) UpdateBandpass(frequency, width float64) {
	b.mode = ModeBandPass
	b.frequency = frequency
	b.width = width

	w0 := 2.0 * math.Pi * frequency / sampleRate
	alpha := math.Sin(w0) * math.Sinh(math.Log(2.0)/2.0*width*w0/math.Sin(w0))

	a0 := 1.0 + alpha
	a1 := -2.0 * math.Cos(w0)
	a2 := 1.0 - alpha
	b0 := alpha
	b1 := 0.0
	b2 := -1.0 * alpha

	// Normalize filter coefficients
	b0 /= a0
	b1 /= a0
	b2 /= a0
	a1 /= a0
	a2 /= a0

	b.b0 = b0
	b.b1 = b1
	b.b2 = b2
	b.a1 = a1
	b.a2 = a2
}

func (lpf *Butterworth) ProcessSample(samples [][2]float64) {
	for i := range samples {
		switch lpf.mode {
		case ModeLowPass:
			if lpf.frequencyEnv != nil {
				v := lpf.frequencyEnv.GetVal(lpf.position)
				lpf.UpdateCutoff(v)
			}
		case ModeBandPass:
			if lpf.frequencyEnv != nil || lpf.widthEnv != nil {
				freq := lpf.frequency
				if lpf.frequencyEnv != nil {
					freq = lpf.frequencyEnv.GetVal(lpf.position)
				}

				width := lpf.width
				if lpf.widthEnv != nil {
					width = lpf.widthEnv.GetVal(lpf.position)
				}

				lpf.UpdateBandpass(freq, width)
			}
		}

		x := samples[i][0]
		y := lpf.b0*x + lpf.b1*lpf.x1 + lpf.b2*lpf.x2 - lpf.a1*lpf.y1 - lpf.a2*lpf.y2

		lpf.x2 = lpf.x1
		lpf.x1 = x
		lpf.y2 = lpf.y1
		lpf.y1 = y

		samples[i][0] = y
		samples[i][1] = y

		lpf.position++
	}
}

func (lpf *Butterworth) GetSetter(k string) func(float64) {
	fmt.Println("get setter on filter: ", k)
	switch lpf.mode {
	case ModeLowPass:
		if k == "cutoff" {
			return func(v float64) {
				lpf.UpdateCutoff(v)
			}
		}
		return nil
	case ModeBandPass:
		switch k {
		case "frequency":
			return func(v float64) {
				lpf.UpdateBandpass(v, lpf.width)
			}
		case "width":
			return func(v float64) {
				lpf.UpdateBandpass(lpf.frequency, v)
			}
		default:
			return nil
		}
	default:
		return nil
	}
}

func (lpf *Butterworth) Process(src beep.Streamer) beep.Streamer {
	return beep.StreamerFunc(func(samples [][2]float64) (n int, ok bool) {
		n, ok = src.Stream(samples)
		for i := range samples[:n] {
			x := samples[i][0]
			y := lpf.b0*x + lpf.b1*lpf.x1 + lpf.b2*lpf.x2 - lpf.a1*lpf.y1 - lpf.a2*lpf.y2

			lpf.x2 = lpf.x1
			lpf.x1 = x
			lpf.y2 = lpf.y1
			lpf.y1 = y

			samples[i][0] = y
			samples[i][1] = y
		}
		return n, ok
	})
}

type MovingAverageFilter struct {
	buffer []float64
	index  int
}

func NewMovingAverageFilter(size int) *MovingAverageFilter {
	return &MovingAverageFilter{
		buffer: make([]float64, size),
		index:  0,
	}
}

func (maf *MovingAverageFilter) Process(src beep.Streamer) beep.Streamer {
	return beep.StreamerFunc(func(samples [][2]float64) (n int, ok bool) {
		n, ok = src.Stream(samples)
		if !ok {
			return n, false
		}

		for i := range samples {
			maf.buffer[maf.index] = samples[i][0]
			maf.index = (maf.index + 1) % len(maf.buffer)

			sum := 0.0
			for _, v := range maf.buffer {
				sum += v
			}
			average := sum / float64(len(maf.buffer))

			samples[i][0] = average
			samples[i][1] = average
		}

		return n, true
	})
}

type Voice struct {
	position int
	paused   bool

	sub beep.Streamer

	envs []*PureEnv

	finalVal [2]float64
}

func (e *Voice) Err() error {
	return nil
}

func (e *Voice) Start() {
	e.paused = false
}

func (e *Voice) Stop() {
	e.paused = true
	for _, pe := range e.envs {
		pe.Stop()
	}
}

func (e *Voice) Silent() bool {
	if !e.paused {
		return false
	}

	for _, env := range e.envs {
		if !env.Done() {
			return false
		}
	}

	return true
}

func (e *Voice) Stream(samples [][2]float64) (int, bool) {
	if e.Silent() {
		if e.finalVal[0] != 0 {
			fade := 50
			if len(samples) < fade {
				fade = len(samples)
			}
			for i := 0; i < fade; i++ {
				samples[i][0] = e.finalVal[0] * float64(fade-i) / float64(fade)
				samples[i][1] = samples[i][0]
			}
			return fade, false
		}
		return 0, false
	}

	n, ok := e.sub.Stream(samples)
	if !ok {
		return n, ok
	}

	if n > 0 {
		e.finalVal = samples[n-1]
	}

	return n, ok
}

func (v *Voice) GetSetter(k string) func(float64) {
	sable, ok := v.sub.(Settable)
	if !ok {
		return nil
	}

	return sable.GetSetter(k)
}

/*
. Peak
.  /\ Dropoff
. /  ----\
./        \ Decay
Zero____________________________________
*/
type PureEnv struct {
	Zero    float64
	Max     float64
	Peak    int
	Dropoff int
	Plateau float64
	Tail    int
	End     float64

	Released   *int
	LastPos    int
	ReleaseVal float64
}

func (e *PureEnv) Stop() {
	e.ReleaseVal = e.getVal(e.LastPos)
	relv := e.LastPos
	e.Released = &relv
}

func (e *PureEnv) Done() bool {
	return e.Released != nil && e.LastPos > *e.Released+e.Tail
}

func (e *PureEnv) GetVal(pos int) float64 {
	e.LastPos = pos

	return e.getVal(pos)
}

func (e *PureEnv) getVal(pos int) float64 {
	if e.Released != nil {
		past := pos - *e.Released
		if past > e.Tail {
			return e.End
		}
		return e.ReleaseVal + ((float64(past) / float64(e.Tail)) * (e.End - e.ReleaseVal))
	}

	if pos < e.Peak {
		rnge := e.Max - e.Zero
		return e.Zero + (rnge * float64(pos) / float64(e.Peak))
	}
	if pos < e.Peak+e.Dropoff {
		return e.Max + ((float64(pos-e.Peak) / float64(e.Dropoff)) * (e.Plateau - e.Max))
	}

	return e.Plateau
}

type FMWave struct {
	sampleRate float64
	frequency  float64
	position   int
	amplitude  float64
	mfreq      float64
	mfreqmod   float64
	impact     float64

	cwfunc OscFunc
	fmfunc OscFunc
}

func (sw *FMWave) Stream(samples [][2]float64) (n int, ok bool) {
	for i := range samples {

		sample := sw.fmfunc(calcPhase(sw.position, sw.sampleRate, sw.mfreq*sw.mfreqmod))

		value := sw.cwfunc(calcPhase(sw.position, sampleRate, sw.frequency) + (sw.impact * sample))

		samples[i][0] = value * sw.amplitude
		samples[i][1] = value * sw.amplitude
		sw.position++
	}
	return len(samples), true
}

func (sw *FMWave) Err() error {
	return nil
}

func (sw *FMWave) GetSetter(k string) func(float64) {
	switch k {
	case "mfreq":
		return func(v float64) {
			sw.mfreq = v
		}
	case "mfreqmod":
		return func(v float64) {
			sw.mfreqmod = v
		}
	default:
		return nil
	}
}

type WaveFolder struct {
	thresh float64
}

func threshInvert(thr, v float64) float64 {
	if v > thr {
		return thr - (v - thr)
	}
	if v < -1*thr {
		return (-1 * thr) - (v + thr)
	}
	return v
}

func threshSqrt(thr, v float64) float64 {
	if v > thr {
		return thr + math.Sqrt(v-thr)
	}
	if v < -1*thr {
		return (-1 * thr) - math.Sqrt(-1*(v+thr))
	}
	return v
}

func (wf *WaveFolder) processVal(v float64) float64 {
	//ov := threshSqrt(wf.thresh, v)
	//return ov
	if v > wf.thresh {
		return wf.thresh - (v - wf.thresh)
	}
	if v < -1*wf.thresh {
		return (-1 * wf.thresh) - (v + wf.thresh)
	}

	return v
}

func (wf *WaveFolder) ProcessSample(samples [][2]float64) {
	for i := range samples {
		v := wf.processVal(samples[i][0])
		samples[i][0] = v
		samples[i][1] = v
	}
}

func (wf *WaveFolder) GetSetter(k string) func(float64) {
	switch k {
	case "threshold":
		return func(v float64) {
			wf.thresh = v
		}
	default:
		return nil
	}

}

type SineWave struct {
	sampleRate float64
	frequency  float64
	position   int
	amplitude  float64
}

func sineOsc(phase float64) float64 {
	return math.Sin(2 * math.Pi * phase)
}

func calcPhase(pos int, samplerate, freq float64) float64 {
	return float64(pos) / samplerate * freq
}

func (sw *SineWave) Stream(samples [][2]float64) (n int, ok bool) {
	for i := range samples {
		value := sineOsc(calcPhase(sw.position, sw.sampleRate, sw.frequency))
		samples[i][0] = value * sw.amplitude
		samples[i][1] = value * sw.amplitude
		sw.position++
	}
	return len(samples), true
}

func (sw *SineWave) Err() error {
	return nil
}

func tanOsc(phase float64) float64 {
	v := math.Tan(2*math.Pi*phase) / 5
	if v > 1 {
		return 1
	}
	if v < -1 {
		return -1
	}
	return v

}

type WeirdSine struct {
	sampleRate   float64
	frequency    float64
	freqEnv      *PureEnv
	position     int
	amplitude    float64
	amplitudeEnv *PureEnv
	freqMod      float64
	fmDecay      float64
}

func (sw *WeirdSine) GetSetter(k string) func(float64) {
	switch k {
	case "freqMod":
		return func(v float64) {
			sw.freqMod = v
		}
	case "fmDecay":
		return func(v float64) {
			sw.fmDecay = v
		}
	case "amp":
		return func(v float64) {
			fmt.Println("setting aplitude: ", v)
			sw.amplitude = v
		}
	default:
		fmt.Println("unknown setting: ", k)
		return nil
	}
}

func (sw *WeirdSine) Stream(samples [][2]float64) (n int, ok bool) {
	//fmt.Println("weird sine: ", sw.amplitudeEnv.GetVal(sw.position))
	for i := range samples {
		freq := sw.frequency
		if sw.freqEnv != nil {
			freq *= sw.freqEnv.GetVal(sw.position)
		}

		value := sineOsc(calcPhase(sw.position, sw.sampleRate, freq+(rand.Float64()*sw.freqMod)))
		amp := sw.amplitude
		if sw.amplitudeEnv != nil {
			amp *= sw.amplitudeEnv.GetVal(sw.position)
		}

		samples[i][0] = value * amp
		samples[i][1] = value * amp
		sw.position++
		if sw.freqMod > 0 {
			//sw.freqMod -= sw.fmDecay
		}
	}
	return len(samples), true
}

func (sw *WeirdSine) Err() error {
	return nil
}

type SawWave struct {
	sampleRate float64
	frequency  float64
	position   int
	amplitude  float64

	phase float64

	bow float64

	amplitudeEnv *PureEnv
}

func sawOsc(ph float64) float64 {
	_, phase := math.Modf(ph)
	return (2 * phase) - 1
}

func (sw *SawWave) Stream(samples [][2]float64) (n int, ok bool) {
	/*
		for i := range samples {
			pos := sw.position % int(sw.frequency)
			value := 1 - (float64(pos) * 2 / float64(sw.frequency))

			samples[i][0] = value * sw.amplitude
			samples[i][1] = value * sw.amplitude
			sw.position++
		}
	*/
	period := sw.sampleRate / sw.frequency

	for i := range samples {
		// Calculate the current phase
		_, sw.phase = math.Modf(float64(sw.position) / period)

		// Generate the sawtooth wave sample
		sample := 2*sw.phase - 1

		if sw.bow > 0 {
			bowadj := 1 - math.Pow(1-sw.phase, 2)
			sample += bowadj
		}

		amp := sw.amplitude
		if sw.amplitudeEnv != nil {
			amp = sw.amplitudeEnv.GetVal(sw.position)
		}

		// Update the output samples
		samples[i][0] = sample * amp
		samples[i][1] = sample * amp
		sw.position++
	}
	return len(samples), true
}

func (sw *SawWave) GetSetter(k string) func(float64) {
	switch k {
	case "bow":
		return func(v float64) {
			fmt.Println("bow set to: ", v)
			sw.bow = v
		}
	default:
		return nil
	}
}

func (sw *SawWave) Err() error {
	return nil
}

type DirtySquare struct {
	sampleRate float64
	frequency  float64
	position   int
	amplitude  float64
	dip        float64

	median float64
}

func (sw *DirtySquare) GetSetter(k string) func(float64) {
	switch k {
	case "median":
		return func(v float64) {
			sw.median = v
		}
	case "dip":
		return func(v float64) {
			sw.dip = v
		}
	default:
		return nil
	}
}

func (sw *DirtySquare) Stream(samples [][2]float64) (n int, ok bool) {
	period := sw.sampleRate / sw.frequency

	for i := range samples {
		phase := math.Mod(float64(sw.position), period)
		var value float64
		if phase < period*sw.median {
			value = 1 - (sw.dip * phase)
		} else {
			value = -1 + (sw.dip * phase)
		}
		samples[i][0] = value * sw.amplitude
		samples[i][1] = value * sw.amplitude
		sw.position++
	}
	return len(samples), true
}

func (sw *DirtySquare) Err() error {
	return nil
}

type SquareWave struct {
	sampleRate float64
	frequency  float64
	position   int
	amplitude  float64
}

func squareOsc(pos int, sampleRate, freq float64) float64 {
	value := math.Sin(2 * math.Pi * float64(pos) / sampleRate * freq)
	if value >= 0 {
		value = 1
	} else {
		value = -1
	}
	return value
}

func (sw *SquareWave) Stream(samples [][2]float64) (n int, ok bool) {
	for i := range samples {
		value := softSquareOsc(sw.position, sw.sampleRate, sw.frequency) * sw.amplitude
		samples[i][0] = value
		samples[i][1] = value
		sw.position++
	}
	return len(samples), true
}

func (sw *SquareWave) Err() error {
	return nil
}

func softSquareOsc(pos int, sampleRate, freq float64) float64 {
	value := math.Sin(2*math.Pi*float64(pos)/sampleRate*freq) * 5
	if value > 1 {
		value = 1
	} else if value < -1 {
		value = -1
	}
	return value
}

type Noise struct {
	amplitude float64
	ampEnv    *PureEnv
	position  int
}

func (nw *Noise) Stream(samples [][2]float64) (n int, ok bool) {
	for i := range samples {
		amp := nw.amplitude
		if nw.ampEnv != nil {
			amp = nw.ampEnv.GetVal(nw.position)
		}
		val := rand.Float64()
		samples[i][0] = val * amp
		samples[i][1] = val * amp
		nw.position++
	}
	return len(samples), true
}

func (nw *Noise) Err() error {
	return nil
}

type NoiseWave struct {
	amplitude  float64
	frequency  float64
	position   int
	sampleRate float64
}

func (nw *NoiseWave) Stream(samples [][2]float64) (n int, ok bool) {
	for i := range samples {
		value := sineOsc(calcPhase(nw.position, nw.sampleRate, nw.frequency))
		val := (1 - (2 * rand.Float64())) * value
		samples[i][0] = val * nw.amplitude
		samples[i][1] = val * nw.amplitude
		nw.position++
	}
	return len(samples), true
}

func (nw *NoiseWave) Err() error {
	return nil
}

type Instrument struct {
	newVoice func(note int64) *Voice

	voiceVals map[string]float64

	activeLk sync.Mutex
	active   []*Voice

	effect Effect

	tmp [512][2]float64

	done bool

	controllerBind func(*Instrument, *MidiController) error
}

func NewInstrument(vf func(int64) *Voice) *Instrument {
	return &Instrument{
		newVoice:  vf,
		voiceVals: make(map[string]float64),
	}
}

func (i *Instrument) Bind(mc *MidiController) error {
	if i.controllerBind == nil {
		mc.SetInst(i)
		return nil
	}
	return i.controllerBind(i, mc)
}

func (i *Instrument) GetSetter(k string) func(float64) {
	i.activeLk.Lock()
	defer i.activeLk.Unlock()

	parts := strings.Split(k, ".")
	sub := strings.Join(parts[1:], ".")
	switch parts[0] {
	case "voice":
		// TODO: could do this more efficiently by storing the fact
		// that we have a setter for a given value, and pulling that
		// from every new voice when we create it
		return func(v float64) {
			i.voiceVals[sub] = v
			for _, act := range i.active {
				ss := act.GetSetter(sub)
				if ss != nil {
					ss(v)
				}
			}
		}
	default:
		return i.effect.GetSetter(k)
	}

}

func (i *Instrument) Play(note int64) func() {
	nv := i.newVoice(note)

	i.activeLk.Lock()
	defer i.activeLk.Unlock()

	for k, v := range i.voiceVals {
		sf := nv.GetSetter(k)
		if sf == nil {
			fmt.Println("no val for: ", k)
			continue
		}
		sf(v)
	}

	i.active = append(i.active, nv)
	nv.Start()

	return nv.Stop
}

func (i *Instrument) Stream(samples [][2]float64) (int, bool) {
	i.activeLk.Lock()
	defer i.activeLk.Unlock()

	osamples := samples

	for i := range samples {
		samples[i] = [2]float64{}
	}

	if i.done {
		return 0, false
	}

	if len(i.active) == 0 {
		if i.effect != nil {
			i.effect.ProcessSample(osamples)
		}
		return len(samples), true
	}

	/*

		var voices []beep.Streamer
		for _, act := range i.active {
			voices = append(voices, act)
		}

		n, _ := beep.Mix(voices...).Stream(samples)
		return n, true
	*/

	rem := make(map[int]bool)
	for len(samples) > 0 && len(rem) < len(i.active) {
		toStream := len(i.tmp)
		if toStream > len(samples) {
			toStream = len(samples)
		}

		// clear the samples
		for i := range samples[:toStream] {
			samples[i] = [2]float64{}
		}

		for ix, st := range i.active {
			// mix the stream
			sn, ok := st.Stream(i.tmp[:toStream])
			if !ok {
				rem[ix] = true
			}

			if sn > 0 {
				//fmt.Println("temp values", sn, i.tmp[0])
			}
			for iv := range i.tmp[:sn] {
				samples[iv][0] += i.tmp[iv][0]
				samples[iv][1] += i.tmp[iv][1]
			}
		}

		samples = samples[toStream:]
	}

	var curs int
	for ix := 0; ix < len(i.active); ix++ {
		if rem[ix] {
			continue
		}

		i.active[curs] = i.active[ix]
		curs++
	}
	i.active = i.active[:curs]

	if i.effect != nil {
		i.effect.ProcessSample(osamples)
	}

	return len(osamples), true
}

func (i *Instrument) Err() error {
	return nil
}

type Stack struct {
	//inst *Instrument

	instruments []*Instrument

	streamer beep.Streamer

	filter *Butterworth

	recorder *Recorder

	//delay *Delay2

	sr beep.SampleRate

	lk sync.Mutex

	activeNotes map[int64]func()
}

func (c *Stack) AddInstrument(inst *Instrument) {
	c.instruments = append(c.instruments, inst)
}

func (c *Stack) Stream(samples [][2]float64) (int, bool) {

	if c.streamer == nil {
		var streamers []beep.Streamer
		for _, i := range c.instruments {
			streamers = append(streamers, i)
		}

		c.streamer = beep.Mix(streamers...)
		//c.streamer = c.filter.Process(c.streamer)
		//out = c.delay.Process(out)

		c.recorder.sub = c.streamer
	}

	n, ok := c.recorder.Stream(samples)
	return n, ok
}

func (c *Stack) Err() error {
	return nil
}

func setupStack(sr beep.SampleRate) *Stack {
	/*
		f := &LowPassFilter{
			cutoffFreq: 200,
			sampleRate: sampleRate,
		}
		f.UpdateCutoff(500)
	*/
	f := NewLowPass(2000, sampleRate)

	recorder := &Recorder{
		buf: make([][2]float64, 10000),
	}

	c := &Stack{
		filter:   f,
		recorder: recorder,
		//delay:       delay,
		sr:          sr,
		activeNotes: make(map[int64]func()),
	}

	return c
}

type AmpControl struct {
	env      *PureEnv
	position int
}

func (ac *AmpControl) GetSetter(k string) func(float64) {
	return nil
}

func (ac *AmpControl) ProcessSample(samples [][2]float64) {
	for i := range samples {
		v := ac.env.GetVal(ac.position)
		samples[i][0] *= v
		samples[i][1] *= v
		ac.position++
	}
}

func snareDrum(note int64) *Voice {
	frequency := 880 * math.Pow(2, (float64(note)-69)/12)
	cenv := &PureEnv{
		Max:     frequency,
		Dropoff: sr.N(time.Millisecond * 100),
		Plateau: 0.0001,
	}

	lpf := NewLowPass(frequency, sampleRate)
	lpf.frequencyEnv = cenv

	ampEnv := &PureEnv{
		Max:     1,
		Dropoff: sr.N(time.Millisecond * 100),
		Plateau: 0,
		Tail:    0,
	}

	out := &Noise{
		amplitude: 1,
	}

	ac := &AmpControl{
		env: ampEnv,
	}
	es := &EffectsStack{}
	es.AddEffect("lowpass", lpf)
	es.AddEffect("amp", ac)

	ew := &EffectWrap{
		sub:    out,
		effect: es,
	}
	return &Voice{
		sub:    ew,
		paused: true,
		envs:   []*PureEnv{cenv, ampEnv},
	}
}

func kickDrum(note int64) *Voice {

	pitchEnv := &PureEnv{
		Max:     0.5,
		Peak:    0,
		Dropoff: sr.N(time.Millisecond * 50),
		Plateau: 0.05,
		Tail:    0,
	}

	volEnv := &PureEnv{
		Max:     1,
		Peak:    0,
		Dropoff: sr.N(time.Millisecond * 100),
		Plateau: 0,
	}

	out := &WeirdSine{
		sampleRate:   sampleRate,
		amplitude:    1,
		freqMod:      0,
		frequency:    440 * math.Pow(2, (float64(note)-69)/12),
		freqEnv:      pitchEnv,
		amplitudeEnv: volEnv,
	}

	return &Voice{
		sub:    out,
		paused: true,
		envs:   []*PureEnv{pitchEnv, volEnv},
	}
}

type EffectWrap struct {
	sub    beep.Streamer
	effect Effect
}

func (ew *EffectWrap) Stream(samples [][2]float64) (int, bool) {
	n, ok := ew.sub.Stream(samples)
	ew.effect.ProcessSample(samples[:n])
	return n, ok
}

func (ew *EffectWrap) Err() error {
	return nil
}

func (ew *EffectWrap) GetSetter(k string) func(float64) {
	ss, ok := ew.sub.(Settable)
	if ok {
		val := ss.GetSetter(k)
		if val != nil {
			return val
		}
	}

	return ew.effect.GetSetter(k)
}

func dirtySq(note int64) *Voice {
	sq := &DirtySquare{
		sampleRate: sampleRate,
		amplitude:  0.1,
		median:     0.2,
		frequency:  440 * math.Pow(2, (float64(note)-69)/12),
	}

	return &Voice{
		sub:    sq,
		paused: true,
		//envs:   []*PureEnv{penv},
	}
}

func freqModBasic(note int64) *Voice {
	fwm := &FMWave{
		sampleRate: sampleRate,
		amplitude:  0.3,
		frequency:  440 * math.Pow(2, (float64(note)-69)/12),
		mfreq:      440 * math.Pow(2, (float64(note)-69)/12),
		mfreqmod:   1,
		impact:     2,
		fmfunc:     sineOsc,
		cwfunc:     sawOsc,
	}

	return &Voice{
		sub:    fwm,
		paused: true,
		//envs:   []*PureEnv{penv},
	}
}

func normalSine(note int64) *Voice {
	sw := &SineWave{
		sampleRate: sampleRate,
		amplitude:  0.3,
		frequency:  440 * math.Pow(2, (float64(note)-69)/12),
	}

	return &Voice{
		sub:    sw,
		paused: true,
		//envs:   []*PureEnv{penv},
	}
}

func foldedSine(note int64) *Voice {
	sw := &SineWave{
		sampleRate: sampleRate,
		amplitude:  0.99,
		frequency:  440 * math.Pow(2, (float64(note)-69)/12),
	}

	wf := &WaveFolder{thresh: 0.6}

	out := &EffectWrap{
		sub:    sw,
		effect: wf,
	}
	return &Voice{
		sub:    out,
		paused: true,
		//envs:   []*PureEnv{penv},
	}
}

func moduNoise(note int64) *Voice {
	noise := &Noise{amplitude: 200}

	frequency := 440 * math.Pow(2, (float64(note)-69)/12)
	filter := NewBandPass(sampleRate, frequency, 0.001)
	//filter.frequencyEnv = penv

	var out beep.StreamerFunc = func(samples [][2]float64) (int, bool) {
		n, ok := noise.Stream(samples)
		filter.ProcessSample(samples[:n])
		return n, ok
	}

	return &Voice{
		sub:    out,
		paused: true,
		//envs:   []*PureEnv{penv},
	}
}

func wahSaw(note int64) *Voice {
	saw := &SawWave{sampleRate: sampleRate, amplitude: 0.2, bow: 0}
	saw.frequency = 880 * math.Pow(2, (float64(note)-69)/12)

	saw2 := &SawWave{sampleRate: sampleRate, amplitude: 0.2, bow: 0.5}
	saw2.frequency = 440 * math.Pow(2, (float64(note)-69)/12)

	sub := &SineWave{sampleRate: sampleRate, amplitude: 0.2}
	sub.frequency = 220 * math.Pow(2, (float64(note)-69)/12)

	penv := &PureEnv{
		Max:     saw.frequency * 1.5,
		Peak:    sr.N(time.Millisecond * 150),
		Dropoff: sr.N(time.Millisecond * 50),
		Plateau: saw.frequency * 1,
		Tail:    sr.N(time.Millisecond * 100),
		//End:     saw.frequency / 2,
	}

	filter := NewBandPass(sampleRate, saw.frequency*1.5, sampleRate)
	filter.frequencyEnv = penv

	mix := beep.Mix(saw, saw2, sub)

	var out beep.StreamerFunc = func(samples [][2]float64) (int, bool) {
		n, ok := mix.Stream(samples)
		filter.ProcessSample(samples[:n])
		return n, ok
	}

	return &Voice{
		sub:    out,
		paused: true,
		//envs:   []*PureEnv{penv},
	}
}

func newVoice(note int64) *Voice {
	sine := &SineWave{sampleRate: sampleRate, amplitude: 0.3}
	sine.frequency = 220 * math.Pow(2, (float64(note)-69)/12)
	square := &SquareWave{sampleRate: sampleRate, amplitude: 0.3}
	square.frequency = 880 * math.Pow(2, (float64(note)-69)/12) * 4
	square2 := &SquareWave{sampleRate: sampleRate, amplitude: 0.3}
	square2.frequency = 220 * math.Pow(2, (float64(note)-69)/12)
	sw := &SawWave{sampleRate: sampleRate, amplitude: 0.3}
	sw.frequency = 880 * math.Pow(2, (float64(note)-69)/12)

	mix := beep.Mix(sine, square, square2, sw)

	fwm := &FMWave{
		sampleRate: sampleRate,
		amplitude:  0.3,
		frequency:  440 * math.Pow(2, (float64(note)-69)/12),
		mfreq:      440 * math.Pow(2, (float64(note)-69)/12),
		mfreqmod:   1,
		impact:     2,
		fmfunc:     sineOsc,
		cwfunc:     sawOsc,
	}

	fwm2 := &FMWave{
		sampleRate: sampleRate,
		amplitude:  0.4,
		frequency:  440 * math.Pow(2, (float64(note)-69)/12),
		mfreq:      55 * math.Pow(2, (float64(note)-69)/12),
		mfreqmod:   1,
		impact:     0.1,
		fmfunc:     sineOsc,
		cwfunc:     sineOsc,
	}

	noise := &NoiseWave{
		amplitude:  0.06,
		frequency:  440 * math.Pow(2, (float64(note)-69)/12),
		sampleRate: sampleRate,
	}
	mix = beep.Mix(fwm, fwm2, sine, noise)
	mix = beep.Mix(fwm2, sine, noise)

	sine1 := &WeirdSine{sampleRate: sampleRate, amplitude: 0.3, freqMod: 0}
	sine1.frequency = 220 * math.Pow(2, (float64(note)-69)/12)
	sine2 := &WeirdSine{sampleRate: sampleRate, amplitude: 0.5, freqMod: 0}
	sine2.frequency = 219.99 * math.Pow(2, (float64(note)-69)/12)
	sine3 := &WeirdSine{sampleRate: sampleRate, amplitude: 0.2, freqMod: 0.8, fmDecay: 1 / float64(sr.N(time.Second*7))}
	sine3.frequency = 220 * math.Pow(2, (float64(note)-69)/12)
	sine7 := &WeirdSine{sampleRate: sampleRate, amplitude: 0.2, freqMod: 1, fmDecay: 1 / float64(sr.N(time.Second))}
	sine7.frequency = 220 * math.Pow(2, (float64(note)-69)/12)
	sine4 := &SineWave{sampleRate: sampleRate, amplitude: 0.1}
	sine4.frequency = 440.1 * math.Pow(2, (float64(note)-69)/12)
	sine5 := &SineWave{sampleRate: sampleRate, amplitude: 0.08}
	sine5.frequency = 879.998 * math.Pow(2, (float64(note)-69)/12)
	sine6 := &SineWave{sampleRate: sampleRate, amplitude: 0.05}
	sine6.frequency = 2 * 880.01 * math.Pow(2, (float64(note)-69)/12)

	mix = beep.Mix(sine1, sine2, sine3, sine4, sine5, sine6, sine7)

	/*
		sine1 := &SineWave{sampleRate: sampleRate, amplitude: 0.3}
		sine1.frequency = 220 * math.Pow(2, (float64(note)-69)/12)
		sine2 := &SineWave{sampleRate: sampleRate, amplitude: 0.3}
		sine2.frequency = 440 * math.Pow(2, (float64(note)-69)/12)
		sine3 := &SineWave{sampleRate: sampleRate, amplitude: 0.3}
		sine3.frequency = 880 * math.Pow(2, (float64(note)-69)/12)

		mix = beep.Mix(sine1, sine2, sine3, noise)
	*/

	//out = sw

	saw := &SawWave{sampleRate: sampleRate, amplitude: 0.2, bow: 0.9}
	saw.frequency = 2 * 880.01 * math.Pow(2, (float64(note)-69)/12)
	mix = saw

	penv := &PureEnv{
		Max:     0.5,
		Peak:    0,
		Dropoff: sr.N(time.Millisecond * 50),
		Plateau: 0.3,
		Tail:    sr.N(time.Millisecond * 50),
	}

	sine2.amplitudeEnv = penv

	mix = beep.Mix(fwm, fwm2, sine, noise)

	mix = MulMix(sine, square)

	mix = saw
	return &Voice{
		sub: mix,
		//sub:     mix,
		paused: true,
		//envs:   []*PureEnv{penv},
	}
}

func playTestNotes() {
	sr := beep.SampleRate(sampleRate)
	speaker.Init(sr, sr.N(time.Second/20))

	done := make(chan bool)

	controller := setupStack(sr)

	speaker.Play(beep.Seq(controller, beep.Callback(func() {
		fmt.Println("DONE")
		done <- true
	})))

	script := []string{
		`vf = func(note) {
			freq = ntf(note)
			sw = sine(freq, 1)
			return newVoice(sw)
		}`,
		`ins = makeinst(vf)`,
		`mc = getMidi()`,
		`mc.SetInst(ins)`,
		//`a = makearp(ins, 2, [60,62,65,68])`,
		//`a.Run()`,
	}

	system := NewSystem(controller)

	for _, c := range script {
		if err := system.ProcessCmd(c); err != nil {
			fmt.Println("ERROR: ", err)
		}
	}

	time.Sleep(time.Second * 10)
}

func linearMap(inmax int, low, hi float64) func(int64) float64 {
	return func(v int64) float64 {
		return low + ((hi - low) * (float64(v) / float64(inmax)))
	}
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	pfi, err := os.Create("cpu.prof")
	if err != nil {
		panic(err)
	}
	pprof.StartCPUProfile(pfi)
	go func() {
		time.Sleep(time.Second * 15)
		fmt.Println("writing profile...")
		pprof.StopCPUProfile()
		pfi.Close()
	}()
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "test":
			playTestNotes()
			return
		case "draw":
			draw()
			return
		}
	}
	portmidi.Initialize()
	defer portmidi.Terminate()

	fmt.Println("device info: ", portmidi.Info(portmidi.DefaultInputDeviceID()))

	mc, err := OpenController(portmidi.DefaultInputDeviceID())
	if err != nil {
		panic(err)
	}

	sr := beep.SampleRate(sampleRate)
	speaker.Init(sr, sr.N(time.Second/20))

	done := make(chan bool)

	controller := setupStack(sr)

	//inst := NewDirtySquare()
	//inst := NewWahSaw()

	inst := NewSpicySaw()
	inst.Bind(mc)
	controller.AddInstrument(inst)

	/*
		fcutoff := inst.GetSetter("filter.cutoff")
		sweeper := &Sweeper{
			Action: func(v float64) {
				val := ((500 * v) + 200)
				fcutoff(val)
			},
		}

		_ = sweeper
		//go sweeper.Run()
	*/

	kick := NewInstrument(kickDrum)
	snare := NewInstrument(snareDrum)
	controller.AddInstrument(kick)
	controller.AddInstrument(snare)

	clk := NewClock(110, 32)

	seq := &Sequencer{
		Notes:    []int64{60, 63, 67, 71},
		NoteSize: 16,
		Inst:     mc.Target,
		corrupt:  true,
	}
	kickseq := &Sequencer{
		Notes:    []int64{60, 0},
		NoteSize: 4,
		Inst:     kick,
	}
	snareseq := &Sequencer{
		Notes:    []int64{0, 80},
		NoteSize: 4,
		Inst:     snare,
	}
	clk.Sequences = append(clk.Sequences, seq, kickseq, snareseq)

	//go clk.Run()

	speaker.Play(beep.Seq(controller, beep.Callback(func() {
		fmt.Println("DONE")
		done <- true
	})))

	renderVis(controller)
}

func attempt1(note int64) *Voice {
	saw := &SawWave{sampleRate: sampleRate, amplitude: 0.5, bow: 0}
	saw.frequency = 440 * math.Pow(2, (float64(note)-69)/12)

	volEnv := &PureEnv{
		Max:     0.3,
		Peak:    sr.N(time.Millisecond * 50),
		Dropoff: sr.N(time.Millisecond * 50),
		Plateau: 0.25,
		Tail:    sr.N(time.Millisecond * 100),
	}

	widthEnv := &PureEnv{
		Zero:    0,
		Max:     0.3,
		Peak:    sr.N(time.Millisecond * 100),
		Dropoff: sr.N(time.Millisecond * 2000),
		Plateau: 0.4,
		Tail:    sr.N(time.Millisecond * 100),
		End:     0.2,
	}

	saw.amplitudeEnv = volEnv

	filter := NewBandPass(sampleRate, saw.frequency*2, 4)
	filter.widthEnv = widthEnv

	wf := &WaveFolder{thresh: 0.25}

	es := &EffectsStack{}
	es.AddEffect("folder", wf)
	es.AddEffect("filter", filter)
	_ = filter
	ew := &EffectWrap{
		sub:    saw,
		effect: es,
	}

	return &Voice{
		sub:    ew,
		paused: true,
		envs: []*PureEnv{
			widthEnv,
			volEnv,
		},
	}
}

func NewSpicySaw() *Instrument {
	inst := NewInstrument(attempt1)
	//filter := NewBandPass(sampleRate, 1000, 400)
	//filter := NewLowPass(1000, sampleRate)

	delay := NewDelay2(time.Millisecond*50, 0)
	des := &EffectsStack{}
	des.AddEffect("lp", NewLowPass(200, sampleRate))
	delay.bufEffect = des

	compressor := NewCompressor(0.5, 1.1, 0.005, 0.0002)

	lpf := NewLowPass(200, sampleRate)
	es := &EffectsStack{}
	es.AddEffect("delay", delay)
	es.AddEffect("filter", lpf)
	es.AddEffect("compressor", compressor)

	inst.effect = es
	inst.controllerBind = func(inst *Instrument, mc *MidiController) error {
		mc.Target = inst
		mc.BindKnob(71, inst.GetSetter("filter.frequency"), linearMap(127, 100, 2000))
		mc.BindKnob(72, inst.GetSetter("filter.width"), linearMap(127, 0, 20))
		mc.BindKnob(73, inst.GetSetter("filter.cutoff"), linearMap(127, 100, 2000))
		mc.BindKnob(74, inst.GetSetter("delay.decay"), func(in int64) float64 {
			return float64(in) / 127
		})

		mc.BindKnob(75, inst.GetSetter("delay.delay"), func(in int64) float64 {
			frac := float64(in) / 127
			dur := time.Duration(float64(time.Second) * frac)
			fmt.Println("delay N", dur)
			return float64(sr.N(dur))
		})
		mc.BindKnob(76, inst.GetSetter("voice.folder.threshold"), linearMap(127, 0, 0.5))
		return nil
	}
	return inst
}

func NewWahSaw() *Instrument {
	inst := NewInstrument(wahSaw)
	//filter := NewBandPass(sampleRate, 1000, 400)
	filter := NewLowPass(1000, sampleRate)
	delay := NewDelay2(time.Millisecond*50, 0.3)
	des := &EffectsStack{}
	des.AddEffect("lp", NewLowPass(200, sampleRate))
	delay.bufEffect = des

	compressor := NewCompressor(0.5, 1.1, 0.001, 0.0001)

	es := &EffectsStack{}
	es.AddEffect("filter", filter)
	es.AddEffect("delay", delay)
	es.AddEffect("compressor", compressor)

	inst.effect = es
	inst.controllerBind = func(inst *Instrument, mc *MidiController) error {
		mc.Target = inst
		mc.BindKnob(71, inst.GetSetter("filter.frequency"), linearMap(127, 100, 2000))
		mc.BindKnob(72, inst.GetSetter("filter.width"), linearMap(127, 0, 20))
		mc.BindKnob(73, inst.GetSetter("filter.cutoff"), linearMap(127, 100, 2000))
		mc.BindKnob(74, inst.GetSetter("delay.decay"), func(in int64) float64 {
			return float64(in) / 127
		})

		mc.BindKnob(75, inst.GetSetter("delay.delay"), func(in int64) float64 {
			frac := float64(in) / 127
			dur := time.Duration(float64(time.Second) * frac)
			fmt.Println("delay N", dur)
			return float64(sr.N(dur))
		})
		return nil
	}
	return inst
}

func NewDirtySquare() *Instrument {
	inst := NewInstrument(dirtySq)
	//filter := NewBandPass(sampleRate, 1000, 400)
	filter := NewLowPass(1000, sampleRate)
	delay := NewDelay2(time.Millisecond*50, 0.3)
	//des := &EffectsStack{}
	//des.AddEffect("lp", NewLowPass(200, sampleRate))
	//delay.bufEffect = des

	compressor := NewCompressor(0.5, 1.1, 0.001, 0.0001)

	es := &EffectsStack{}
	es.AddEffect("filter", filter)
	es.AddEffect("delay", delay)
	es.AddEffect("compressor", compressor)

	inst.effect = es
	inst.controllerBind = func(inst *Instrument, mc *MidiController) error {
		mc.Target = inst
		mc.BindKnob(70, inst.GetSetter("voice.mfreqmod"), func(in int64) float64 {
			return math.Pow((float64(in)/127)+0.5, 5)
		})
		mc.BindKnob(71, inst.GetSetter("filter.frequency"), linearMap(127, 100, 2000))
		mc.BindKnob(72, inst.GetSetter("filter.width"), linearMap(127, 0, 20))
		mc.BindKnob(73, inst.GetSetter("filter.cutoff"), linearMap(127, 100, 2000))

		mc.BindKnob(74, inst.GetSetter("delay.decay"), func(in int64) float64 {
			return float64(in) / 127
		})

		mc.BindKnob(75, inst.GetSetter("delay.delay"), func(in int64) float64 {
			frac := float64(in) / 127
			dur := time.Duration(float64(time.Second) * frac)
			fmt.Println("delay N", dur)
			return float64(sr.N(dur))
		})
		mc.BindKnob(76, inst.GetSetter("voice.threshold"), func(in int64) float64 {
			frac := float64(in) / 127
			return frac
		})
		mc.BindKnob(76, inst.GetSetter("voice.median"), func(in int64) float64 {
			frac := float64(in) / 127
			return frac
		})
		mc.BindKnob(77, inst.GetSetter("voice.dip"), func(in int64) float64 {
			frac := float64(in) / 127
			return (frac - 0.5) / 50
		})
		return nil
	}
	return inst
}

var vals = []string{
	"C",
	"C#",
	"D",
	"Eb",
	"E",
	"F",
	"F#",
	"G",
	"G#",
	"A",
	"Bb",
	"B",
}

func noteToString(note int64) string {
	return vals[note%int64(len(vals))]
}

func checkSmooth(vals [][2]float64) {
	cur := float64(0)
	var dir int
	for i := 0; i < len(vals); i++ {
		if vals[i][0] > cur && dir <= 0 {
			fmt.Printf("Change dir up at: %d\t(%f)\n", i, vals[i][0])
			dir = 1
		} else if vals[i][0] < cur && dir >= 0 {
			fmt.Printf("Change dir down at: %d\t(%f)\n", i, vals[i][0])
			dir = -1
		}
		cur = vals[i][0]
	}
}
