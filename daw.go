package main

import (
	"fmt"
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
	barInterval := time.Minute / (time.Duration(c.BPM) * time.Duration(c.MinDiv) * 4)

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

	curnoteStop func()

	cur int
}

func (s *Sequencer) Tick(notesize, pos int) {
	if notesize <= s.NoteSize {
		fmt.Println("tick: ", notesize, pos)
		if s.curnoteStop != nil {
			s.curnoteStop()
		}
		//s.curnoteStop = s.Inst.Play(s.Notes[s.cur%len(s.Notes)])
		s.cur++
	}
}
