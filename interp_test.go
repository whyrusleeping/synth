package main

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/gopxl/beep"
	"github.com/gopxl/beep/wav"
)

func TestInterpreter(t *testing.T) {
	c := new(Stack)
	s := NewSystem(c)

	err := s.ProcessCmd("a = makearp(makeinst(defvoice()), 8, [1,2,3,4,5])")
	if err != nil {
		t.Fatal(err)
	}
}

func TestScript(t *testing.T) {
	c := new(Stack)
	s := NewSystem(c)

	err := s.ProcessCmd(`a = func(a, b, c) {
print(a)
print(b)
print(c)
d = 7
print(d)
}`)
	if err != nil {
		t.Fatal(err)
	}
	err = s.ProcessCmd(`a(4,5,6)`)
	if err != nil {
		t.Fatal(err)
	}
}

/*
func TestClock(t *testing.T) {
	c := NewClock(120, 4)
	seq := &Sequencer{
		Notes:    []int64{1, 2, 3, 4},
		NoteSize: 4,
	}
	c.Sequences = append(c.Sequences, seq)

	c.Run()
}
*/

func TestDrawWave(t *testing.T) {

	inst := NewInstrument(dirtySq)

	fi, err := os.Create("output.wav")
	if err != nil {
		t.Fatal(err)
	}

	var done bool
	go func() {
		time.Sleep(time.Millisecond)
		stop := inst.Play(65)
		time.Sleep(time.Millisecond * 50)
		stop()
		time.Sleep(time.Millisecond * 50)
		done = true
		inst.done = true
	}()
	_ = done

	var n int
	var stopfn func()
	var playednote bool
	var outv []float64
	var out beep.StreamerFunc = func(samples [][2]float64) (int, bool) {
		if !playednote {
			stopfn = inst.Play(65)
			playednote = true
		}
		if n > sampleRate/180 && stopfn != nil {
			stopfn()
			stopfn = nil
		} else if n > sampleRate/120 {
			fmt.Println("done time", n)
			inst.done = true
		}

		on, ok := inst.Stream(samples)
		n += on
		for i := range samples[:on] {
			outv = append(outv, samples[i][0])
		}
		return on, ok
	}

	/*
		var out [][2]float64
		for !done {
			n, ok := inst.Stream(buf)
			out = append(out, buf[:n]...)
			if !ok {
				break
			}
		}
	*/

	//ws := &copier{data: out}

	if err := wav.Encode(fi, out, beep.Format{
		SampleRate: sampleRate,

		NumChannels: 1,

		Precision: 1,
	}); err != nil {
		t.Fatal(err)
	}

	fmt.Println(outv)
}

type copier struct {
	data [][2]float64
	n    int
}

func (s *copier) Stream(samples [][2]float64) (int, bool) {
	subs := s.data[s.n:]
	var n int
	for i := 0; i < len(subs) && i < len(samples); i++ {
		samples[i] = subs[i]
		n++
	}

	s.n += n

	return n, len(s.data[s.n:]) > 0

}

func (s *copier) Err() error {
	return nil
}
