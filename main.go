package main

import (
	"fmt"
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

type LoudnessFixer struct {
	thresh float64

	mul float64

	speed float64
}

func (lf *LoudnessFixer) GetSetter(k string) func(float64) {
	return nil
}

func (lf *LoudnessFixer) ProcessSample(samples [][2]float64) {
	var maxval float64
	for _, v := range samples {
		av := math.Abs(v[0])
		if av > maxval {
			maxval = av
		}
	}

	for i := range samples {
		if lf.mul*maxval < lf.thresh {
			lf.mul *= lf.speed
		} else if lf.mul*maxval > lf.thresh {
			lf.mul /= lf.speed
		}

		samples[i][0] *= lf.mul
		samples[i][1] *= lf.mul
	}
}

type LowPassFilter struct {
	prevRaw      float64
	prevFiltered float64

	cutoffFreq float64
	sampleRate float64
	alpha      float64
}

func (lpf *LowPassFilter) Process(src beep.Streamer) beep.Streamer {
	return beep.StreamerFunc(func(samples [][2]float64) (n int, ok bool) {
		n, ok = src.Stream(samples)

		for i := range samples[:n] {
			nv := lpf.ProcessSample(samples[i][0])
			lpf.prevRaw = samples[i][0]
			lpf.prevFiltered = nv
			samples[i][0] = nv
			samples[i][1] = nv
		}

		return n, ok
	})
}

func (lpf *LowPassFilter) UpdateCutoff(cutoff float64) {
	lpf.cutoffFreq = cutoff
	rc := 1.0 / (2 * math.Pi * lpf.cutoffFreq)
	dt := 1.0 / lpf.sampleRate
	lpf.alpha = dt / (rc + dt)
}

func (lpf *LowPassFilter) ProcessSample(cur float64) float64 {
	fv := lpf.alpha*cur + (1-lpf.alpha)*lpf.prevFiltered
	return math.Max(-1, math.Min(1, fv))
}

type SimpleLowPass struct {
	sampleRate float64
	Cutoff     float64
	LastVal    float64
}

func (slp *SimpleLowPass) ProcessSample(samples [][2]float64) {
	n := len(samples)
	waveLength := float64(n) / slp.sampleRate
	rc := 1.0 / (2 * math.Pi * slp.Cutoff)
	alpha := waveLength / (rc + waveLength)
	for i := range samples {
		samples[i][0] = slp.LastVal + alpha*(samples[i][0]-slp.LastVal)
		slp.LastVal = samples[i][0]
		samples[i][1] = samples[i][0]
	}
}

func (slp *SimpleLowPass) GetSetter(k string) func(float64) {
	return nil
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

	cutoffEnv *PureEnv
}

func NewButterworth(cutoffFreq, sampleRate float64) *Butterworth {
	b := &Butterworth{
		sampleRate: sampleRate,
	}

	b.UpdateCutoff(cutoffFreq)
	return b
}

func (b *Butterworth) UpdateCutoff(cutoffFreq float64) {
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

func (lpf *Butterworth) ProcessSample(samples [][2]float64) {
	for i := range samples {
		if lpf.cutoffEnv != nil {
			v := lpf.cutoffEnv.GetVal(lpf.position)
			lpf.UpdateCutoff(v)
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
	if k == "cutoff" {
		return func(v float64) {
			lpf.UpdateCutoff(v)
		}
	}
	return nil

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
		return 0, false
	}

	n, ok := e.sub.Stream(samples)
	if !ok {
		return n, ok
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
____________________________________
*/
type PureEnv struct {
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
		return e.Max * float64(pos) / float64(e.Peak)
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

	for i := range samples {
		// Calculate the current phase
		_, sw.phase = math.Modf(sw.phase + sw.frequency/sw.sampleRate)

		// Generate the sawtooth wave sample
		sample := 2*sw.phase - 1

		if sw.bow > 0 {
			bowadj := 1 - math.Pow(1-sw.phase, 2)
			sample += bowadj
		}

		// Update the output samples
		samples[i][0] = sample * sw.amplitude
		samples[i][1] = sample * sw.amplitude
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
}

func (nw *Noise) Stream(samples [][2]float64) (n int, ok bool) {
	for i := range samples {
		val := rand.Float64()
		samples[i][0] = val * nw.amplitude
		samples[i][1] = val * nw.amplitude
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

	activeLk sync.Mutex
	active   []*Voice

	effect Effect

	tmp [512][2]float64
}

func (i *Instrument) GetSetter(k string) func(float64) {
	parts := strings.Split(k, ".")
	sub := strings.Join(parts[1:], ".")
	switch parts[0] {
	case "voice":
		// TODO: could do this more efficiently by storing the fact
		// that we have a setter for a given value, and pulling that
		// from every new voice when we create it
		return func(v float64) {
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
	inst *Instrument

	instruments []*Instrument

	streamer beep.Streamer

	filter *Butterworth

	recorder *Recorder

	delay *Delay

	sr beep.SampleRate

	lk sync.Mutex

	activeNotes map[int64]func()
}

func (c *Stack) AddInstrument(inst *Instrument) {
	c.instruments = append(c.instruments, inst)
}

func (c *Stack) StartNote(note int64) {
	c.lk.Lock()
	defer c.lk.Unlock()

	oldstop, ok := c.activeNotes[note]
	if ok {
		fmt.Println("play called with active note")
		oldstop()
	}

	stopf := c.inst.Play(note)

	c.activeNotes[note] = stopf
}

func (c *Stack) StopNote(note int64) {
	c.lk.Lock()
	defer c.lk.Unlock()

	stopf, ok := c.activeNotes[note]
	if !ok {
		return
	}

	stopf()
	delete(c.activeNotes, note)
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
	f := NewButterworth(2000, sampleRate)

	n := sr.N(time.Millisecond * 50)
	delay := &Delay{
		buf:   make([][2]float64, n*10),
		delay: n,
		decay: 0.7,
	}

	recorder := &Recorder{
		buf: make([][2]float64, 10000),
	}

	c := &Stack{
		inst:        &Instrument{newVoice: wahSaw},
		filter:      f,
		recorder:    recorder,
		delay:       delay,
		sr:          sr,
		activeNotes: make(map[int64]func()),
	}

	c.AddInstrument(c.inst)
	return c
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
		amplitude:    0.5,
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

func wahSaw(note int64) *Voice {
	saw := &SawWave{sampleRate: sampleRate, amplitude: 0.2, bow: 0.9}
	saw.frequency = 880 * math.Pow(2, (float64(note)-69)/12)

	penv := &PureEnv{
		Max:     saw.frequency * 1.5,
		Peak:    sr.N(time.Millisecond * 100),
		Dropoff: sr.N(time.Millisecond * 50),
		Plateau: saw.frequency * 1,
		Tail:    sr.N(time.Millisecond * 100),
		End:     saw.frequency / 2,
	}

	filter := NewButterworth(saw.frequency*1.5, sampleRate)
	filter.cutoffEnv = penv
	_ = filter

	var out beep.StreamerFunc = func(samples [][2]float64) (int, bool) {
		n, ok := saw.Stream(samples)
		//filter.ProcessSample(samples[:n])
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

func main() {
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

	mc.Target = controller.inst

	filter := NewButterworth(3000, sampleRate)
	delaySamples := sr.N(time.Millisecond * 200)
	delay := &Delay{
		buf:   make([][2]float64, delaySamples*10),
		delay: delaySamples,
		decay: 0.7,
	}

	//compressor := NewCompressor(float64(sr), 15, 0.5, 0.1, 0.1, 0.5)

	es := &EffectsStack{}
	es.AddEffect("filter", filter)
	es.AddEffect("delay", delay)
	//es.AddEffect("compressor", compressor)

	mc.Target.effect = es

	mc.BindKnob(70, mc.Target.GetSetter("voice.mfreqmod"), func(in int64) float64 {
		return math.Pow((float64(in)/127)+0.5, 5)
	})
	mc.BindKnob(71, mc.Target.GetSetter("filter.cutoff"), func(in int64) float64 {
		// for some reason the filter breaks if we set it above 22k
		return 100 + (1500 * (float64(in) / 127))
	})

	mc.BindKnob(74, mc.Target.GetSetter("delay.decay"), func(in int64) float64 {
		return float64(in) / 127
	})

	mc.BindKnob(75, mc.Target.GetSetter("delay.delay"), func(in int64) float64 {
		frac := float64(in) / 127
		dur := time.Duration(float64(time.Second) * frac)
		fmt.Println("delay N", dur)
		return float64(sr.N(dur))
	})
	mc.BindKnob(76, mc.Target.GetSetter("voice.bow"), func(in int64) float64 {
		frac := float64(in) / 127
		return frac
	})

	fcutoff := mc.Target.GetSetter("filter.cutoff")
	sweeper := &Sweeper{
		Action: func(v float64) {
			val := ((500 * v) + 200)
			fcutoff(val)
		},
	}

	_ = sweeper
	//go sweeper.Run()

	speaker.Play(beep.Seq(controller, beep.Callback(func() {
		fmt.Println("DONE")
		done <- true
	})))

	<-done
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

const (
	screenWidth  = 1000
	screenHeight = 600
	graphWidth   = 800
	graphHeight  = 400
	graphOffsetX = 100
	graphOffsetY = 100
)

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
