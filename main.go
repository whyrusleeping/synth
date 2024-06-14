package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"sync"
	"time"

	"github.com/gopxl/beep"
	"github.com/gopxl/beep/speaker"
	"github.com/maddyblue/go-dsp/fft"
	"github.com/rakyll/portmidi"
	"github.com/veandco/go-sdl2/sdl"
)

type OscFunc func(float64) float64

const (
	sampleRate = 44100
	duration   = 2
)

type Recorder struct {
	lk       sync.Mutex
	buf      [][2]float64
	position int

	sub beep.Streamer
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

type Delay struct {
	buf      [][2]float64
	delay    int
	decay    float64
	position int
}

func (d *Delay) Process(src beep.Streamer) beep.Streamer {
	return beep.StreamerFunc(func(samples [][2]float64) (n int, ok bool) {
		n, ok = src.Stream(samples)

		for i := range samples[:n] {
			samples[i][0] += d.buf[d.position%len(d.buf)][0]
			samples[i][1] += d.buf[d.position%len(d.buf)][1]

			dpos := (d.delay + d.position) % len(d.buf)
			d.buf[dpos][0] = samples[i][0] * d.decay
			d.buf[dpos][1] = samples[i][1] * d.decay

			/*
				rv := rand.Intn(2)
				for i := 0; i < 4; i++ {
					dposr := (d.delay + d.position + rv - 3 + i) % len(d.buf)
					d.buf[dposr][0] += samples[i][0] * d.decay * rand.Float64() * 0.1
					d.buf[dposr][1] += samples[i][1] * d.decay * rand.Float64() * 0.1
				}
			*/

			d.position++
		}

		return n, ok
	})

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

// butterworth filter
type Butterworth struct {
	b0, b1, b2, a1, a2 float64
	x1, x2, y1, y2     float64
	sampleRate         float64
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

type Envelope struct {
	position int
	delay    int
	duration int
	paused   bool

	velocity float64

	maxgain float64

	sub beep.Streamer
}

func (e *Envelope) Err() error {
	return nil
}

func (e *Envelope) Start() {
	e.paused = false
}

func (e *Envelope) Stop() {
	e.paused = true
}

func (e *Envelope) Stream(samples [][2]float64) (int, bool) {
	n, ok := e.sub.Stream(samples)
	if !ok {
		return n, ok
	}

	for i := range samples {
		if e.paused {
			e.velocity -= (e.maxgain / float64(e.delay))
			if e.velocity < 0 {
				e.velocity = 0
			}
		} else {
			e.velocity += (e.maxgain / float64(e.delay))
			if e.velocity > e.maxgain {
				e.velocity = e.maxgain
			}
		}

		samples[i][0] *= e.velocity
		samples[i][1] *= e.velocity

		/*
			if samples[i][0] > 1 || samples[i][0] < -1 {
				fmt.Println("too loud!", samples[i][0])
			}
		*/
		e.position++
	}
	return n, ok
}

type FMWave struct {
	sampleRate float64
	frequency  float64
	position   int
	amplitude  float64
	mfreq      float64
	impact     float64

	cwfunc OscFunc
	fmfunc OscFunc
}

func (sw *FMWave) Stream(samples [][2]float64) (n int, ok bool) {
	for i := range samples {

		// test
		//_, phase := math.Modf(float64(sw.position) * sw.frequency / sw.sampleRate)

		// Generate the sawtooth wave sample
		//sample := 2*phase - 1

		sample := sw.fmfunc(calcPhase(sw.position, sw.sampleRate, sw.mfreq))

		//

		//value := math.Sin(2 * math.Pi * float64(pos) / samplerate * freq)

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

type SawWave struct {
	sampleRate float64
	frequency  float64
	position   int
	amplitude  float64

	phase float64
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

		// Update the output samples
		samples[i][0] = sample
		samples[i][1] = sample
	}
	return len(samples), true
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

type Controller struct {
	voices map[int64]*Envelope

	filter *Butterworth

	recorder *Recorder

	delay *Delay

	sr beep.SampleRate
}

func (c *Controller) StartNote(note int64) {
	v, ok := c.voices[note]
	if !ok {
		v = c.newVoice(note)
		c.voices[note] = v
	}

	v.Start()
}

func (c *Controller) StopNote(note int64) {
	v := c.voices[note]
	v.Stop()
}

func (c *Controller) Stream(samples [][2]float64) (int, bool) {
	if len(c.voices) == 0 {
		for i := range samples {
			samples[i][0] = 0
			samples[i][1] = 0
		}
		return len(samples), true
	}

	var streams []beep.Streamer
	for _, v := range c.voices {
		streams = append(streams, v)
	}

	mixed := beep.Mix(streams...)
	mixed = c.filter.Process(mixed)
	//mixed = c.delay.Process(mixed)

	c.recorder.sub = mixed

	return c.recorder.Stream(samples)
}

func (c *Controller) Err() error {
	return nil
}

func setupController(sr beep.SampleRate) *Controller {
	/*
		f := &LowPassFilter{
			cutoffFreq: 200,
			sampleRate: sampleRate,
		}
		f.UpdateCutoff(500)
	*/
	f := NewButterworth(1000, sampleRate)

	n := sr.N(time.Millisecond * 50)
	delay := &Delay{
		buf:   make([][2]float64, n*2),
		delay: n,
		decay: 0.7,
	}

	recorder := &Recorder{
		buf: make([][2]float64, 10000),
	}

	return &Controller{
		voices:   make(map[int64]*Envelope),
		filter:   f,
		recorder: recorder,
		delay:    delay,
		sr:       sr,
	}
}

func (c *Controller) newVoice(note int64) *Envelope {
	sine := &SineWave{sampleRate: sampleRate, amplitude: 0.3}
	square := &SquareWave{sampleRate: sampleRate, amplitude: 0.3}
	sine.frequency = 220 * math.Pow(2, (float64(note)-69)/12)
	square.frequency = 880 * math.Pow(2, (float64(note)-69)/12)
	square2 := &SquareWave{sampleRate: sampleRate, amplitude: 0.3}
	square2.frequency = 220 * math.Pow(2, (float64(note)-69)/12)
	sw := &SawWave{sampleRate: sampleRate, amplitude: 0.3}
	sw.frequency = 880 * math.Pow(2, (float64(note)-69)/12)

	mix := beep.Mix(sine, square, square2, sw)

	_ = mix
	//mix = c.filter.Process(mix)

	// TODO: delay needs to go over the envelope

	//maf := NewMovingAverageFilter(20)
	//mix = maf.Process(mix)

	fwm := &FMWave{
		sampleRate: sampleRate,
		amplitude:  0.3,
		frequency:  440 * math.Pow(2, (float64(note)-69)/12),
		mfreq:      440 * math.Pow(2, (float64(note)-69)/12),
		impact:     2,
		fmfunc:     sineOsc,
		cwfunc:     sawOsc,
	}

	fwm2 := &FMWave{
		sampleRate: sampleRate,
		amplitude:  0.3,
		frequency:  220 * math.Pow(2, (float64(note)-69)/12),
		mfreq:      440 * math.Pow(2, (float64(note)-69)/12),
		impact:     6,
		fmfunc:     sineOsc,
		cwfunc:     sineOsc,
	}

	mix = beep.Mix(fwm, fwm2, sine)
	mix = square2

	//out = c.filter.Process(out)
	//out = sw

	return &Envelope{
		delay: c.sr.N(time.Second / 50),
		sub:   mix,
		//sub:     mix,
		paused:  true,
		maxgain: 0.3,
	}
}

func playTestNotes() {
	sr := beep.SampleRate(sampleRate)
	speaker.Init(sr, sr.N(time.Second/10))

	done := make(chan bool)

	controller := setupController(sr)

	speaker.Play(beep.Seq(controller, beep.Callback(func() {
		fmt.Println("DONE")
		done <- true
	})))

	//controller.StartNote(60)
	//time.Sleep(time.Millisecond * 100)
	//controller.StopNote(60)
	//time.Sleep(time.Second * 3)

	controller.StartNote(60)
	time.Sleep(time.Millisecond * 50)
	controller.StartNote(64)
	time.Sleep(time.Millisecond * 50)
	controller.StartNote(69)
	time.Sleep(time.Second * 2)
	controller.StopNote(60)
	controller.StopNote(64)
	controller.StopNote(69)
	time.Sleep(time.Millisecond * 500)
}

func main() {
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

	in, err := portmidi.NewInputStream(portmidi.DefaultInputDeviceID(), 1024)
	if err != nil {
		panic(err)
	}
	defer in.Close()

	sr := beep.SampleRate(sampleRate)
	speaker.Init(sr, sr.N(time.Second/10))

	done := make(chan bool)

	controller := setupController(sr)

	speaker.Play(beep.Seq(controller, beep.Callback(func() {
		fmt.Println("DONE")
		done <- true
	})))

	for {
		events, err := in.Read(1024)
		if err != nil {
			panic(err)
		}

		for _, event := range events {
			switch event.Status {
			case 0x90:
				note := int64(event.Data1)
				controller.StartNote(note)
			case 0x80:
				note := int64(event.Data1)
				controller.StopNote(note)
			case 0xb0:
				// twisty knobs

				//controller.filter.UpdateCutoff(10000 * (float64(event.Data2) / 128))
				controller.filter.UpdateCutoff(math.Pow(float64(event.Data2), 1.5))

			default:
				b, err := json.Marshal(event)
				if err != nil {
					panic(err)
				}
				fmt.Println(string(b))
			}
		}
	}

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

type Arp struct {
	notes    []int64
	duration time.Duration

	c *Controller
}

func (a *Arp) Run() {
	for {
		for i := 0; i < len(a.notes); i++ {
			a.c.StartNote(a.notes[i])
			time.Sleep(a.duration)
			a.c.StopNote(a.notes[i])
		}
	}
}

func draw() {
	// Initialize SDL
	if err := sdl.Init(sdl.INIT_EVERYTHING); err != nil {
		fmt.Println("Failed to initialize SDL:", err)
		os.Exit(1)
	}
	defer sdl.Quit()

	// Create the window
	window, err := sdl.CreateWindow("Graph", sdl.WINDOWPOS_UNDEFINED, sdl.WINDOWPOS_UNDEFINED, screenWidth, screenHeight, sdl.WINDOW_SHOWN)
	if err != nil {
		fmt.Println("Failed to create window:", err)
		os.Exit(1)
	}
	defer window.Destroy()

	// Create the renderer
	renderer, err := sdl.CreateRenderer(window, -1, sdl.RENDERER_ACCELERATED)
	if err != nil {
		fmt.Println("Failed to create renderer:", err)
		os.Exit(1)
	}
	defer renderer.Destroy()

	sr := beep.SampleRate(sampleRate)
	c := setupController(sr)

	speaker.Init(sr, sr.N(time.Second/20))

	speaker.Play(beep.Seq(c, beep.Callback(func() {
		fmt.Println("DONE")
	})))

	buf := make([][2]float64, 2000)

	keystates := make(map[int]bool)
	dataPoints := make([]float64, len(buf))

	a := &Arp{
		notes:    []int64{60, 64, 67, 72},
		duration: time.Millisecond * 400,
		c:        c,
	}

	go a.Run()

	// Game loop
	running := true
	var octaveAdjust int64
	for running {
		for event := sdl.PollEvent(); event != nil; event = sdl.PollEvent() {
			switch event := event.(type) {
			case *sdl.QuitEvent:
				running = false
			case *sdl.KeyboardEvent:
				var note int64
				switch event.Keysym.Sym {
				case sdl.K_a:
					note = 60
				case sdl.K_s:
					note = 62
				case sdl.K_d:
					note = 64
				case sdl.K_f:
					note = 65
				case sdl.K_g:
					note = 67
				case sdl.K_h:
					note = 69
				case sdl.K_j:
					note = 71
				case sdl.K_k:
					note = 72
				case sdl.K_l:
					note = 74
				}
				note += octaveAdjust

				if event.Type == sdl.KEYUP {
					v := int(event.Keysym.Sym)
					if keystates[v] {
						delete(keystates, int(event.Keysym.Sym))
						if note > 0 {
							c.StopNote(note)
						} else {
							switch v {
							case sdl.K_z:
								octaveAdjust -= 12
							case sdl.K_x:
								octaveAdjust += 12
							}
						}
					}
				} else if event.Type == sdl.KEYDOWN {
					if keystates[int(event.Keysym.Sym)] {
						continue
					}
					keystates[int(event.Keysym.Sym)] = true
					if note > 0 {
						c.StartNote(note)
					}
				}
			}
		}

		c.recorder.GetSnapshot(buf)
		for i, v := range buf {
			dataPoints[i] = v[0]
		}

		fftResult := fft.FFTReal(dataPoints)

		// Get the magnitude spectrum
		magnitudeSpectrum := make([]float64, len(fftResult)/2+1)
		for i, c := range fftResult[:len(magnitudeSpectrum)] {
			magnitudeSpectrum[i] = cmplx.Abs(c) / float64(len(dataPoints))
		}

		// Clear the renderer
		renderer.SetDrawColor(255, 255, 255, 255)
		renderer.Clear()

		graphData(renderer, dataPoints[:500], 50, 50, 600, 200, -1, 1)
		graphData(renderer, magnitudeSpectrum[:100], 50, 300, 600, 200, 0, 0.5)

		// Present the renderer
		renderer.Present()
	}
}

func graphData(renderer *sdl.Renderer, dataPoints []float64, x, y, width, height int32, minval, maxval float64) {
	// Draw the graph axes
	renderer.SetDrawColor(0, 0, 0, 255)
	renderer.DrawLine(x, y+height/2, x+width, y+height/2)
	renderer.DrawLine(x, y, x, y+height)

	spread := maxval - minval
	// Draw the data points
	renderer.SetDrawColor(255, 0, 0, 255)
	for i := 0; i < len(dataPoints)-1; i++ {
		x1 := x + int32(float64(i)*float64(width)/float64(len(dataPoints)-1))
		y1 := y + height - int32((float64(dataPoints[i]-minval)/maxval)*float64(height)/spread)
		x2 := x + int32(float64(i+1)*float64(width)/float64(len(dataPoints)-1))
		//y2 := y + height/2 - int32(dataPoints[i+1]*float64(height)/4)
		y2 := y + height - int32((float64(dataPoints[i+1]-minval)/maxval)*float64(height)/spread)
		renderer.DrawLine(x1, y1, x2, y2)
	}

}
