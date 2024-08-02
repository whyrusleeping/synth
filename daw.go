package main

import (
	"math/rand"
	"time"
)

type Clock struct {
	BPM int
	//NotesPerBar int
	MinDiv int

	Sequences []Seq
}

func NewClock(bpm int, mindiv int) *Clock {
	return &Clock{
		BPM:    bpm,
		MinDiv: mindiv,
	}
}

type Seq interface {
	Tick(int, int)
}

func posToNote(pos int) int {
	if pos%2 == 1 {
		return 32
	}
	if pos%4 == 2 {
		return 16
	}
	if pos%32 == 0 {
		return 1
	}
	if pos%8 == 4 {
		return 8
	}
	if pos%16 == 8 {
		return 4
	}
	if pos%32 == 16 {
		return 2
	}

	return -1
}

func (c *Clock) Run() {
	barInterval := (4 * time.Minute) / (time.Duration(c.BPM) * time.Duration(c.MinDiv))

	var pos int
	for range time.Tick(barInterval) {
		noteSize := posToNote(pos)

		for _, s := range c.Sequences {
			s.Tick(noteSize, pos)
		}
		pos++
	}
}

type Sequencer struct {
	Notes    []int64
	NoteSize int
	Inst     *Instrument

	corrupt bool
	playing []bool

	curnoteStop func()

	cur int
}

func (s *Sequencer) Tick(notesize, pos int) {
	if s.playing == nil {
		s.playing = make([]bool, len(s.Notes))
		for i := range s.playing {
			s.playing[i] = true
		}
	}
	if notesize <= s.NoteSize {
		if s.corrupt && rand.Intn(4) == 0 {
			ix := rand.Intn(len(s.playing))
			s.playing[ix] = !s.playing[ix]
		}
		if s.curnoteStop != nil {
			s.curnoteStop()
		}
		ni := s.cur % len(s.Notes)
		if s.playing[ni] {
			s.curnoteStop = s.Inst.Play(s.Notes[ni])
		}
		s.cur++
	}
}
