package main

import (
	"fmt"
	"math/cmplx"
	"os"
	"time"

	"github.com/c-bata/go-prompt"
	"github.com/gopxl/beep"
	"github.com/gopxl/beep/speaker"
	"github.com/maddyblue/go-dsp/fft"
	"github.com/veandco/go-sdl2/sdl"
)

const (
	screenWidth  = 1000
	screenHeight = 600
	graphWidth   = 800
	graphHeight  = 400
	graphOffsetX = 100
	graphOffsetY = 100
)

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

	c := setupStack(sr)

	speaker.Init(sr, sr.N(time.Second/20))

	speaker.Play(beep.Seq(c, beep.Callback(func() {
		fmt.Println("DONE")
	})))

	buf := make([][2]float64, 2000)

	keystates := make(map[int]bool)
	dataPoints := make([]float64, len(buf))

	/*
		a := &Arp{
			notes:    []int64{60, 64, 67, 72},
			duration: time.Millisecond * 400,
			c:        c,
		}

		go a.Run()
	*/

	sys := NewSystem(c)

	completer := func(d prompt.Document) []prompt.Suggest {
		return nil
	}

	go func() {
		return
		for {
			t := prompt.Input("> ", completer)
			if t == "exit" {
				return
			}
			if err := sys.ProcessCmd(t); err != nil {
				fmt.Println("ERROR: ", err)
			}
		}
	}()

	//select {}

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

		if len(keystates) > 0 {
			c.recorder.GetSnapshot(buf)
		}

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
