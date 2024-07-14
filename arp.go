package main

import (
	"sync"
	"time"
)

type Arp struct {
	notes    []int64
	duration time.Duration

	inst *Instrument
	lk   sync.Mutex
}

func (a *Arp) getInst() *Instrument {
	a.lk.Lock()
	defer a.lk.Unlock()
	return a.inst
}

func (a *Arp) SetInst(inst *Instrument) {
	a.lk.Lock()
	defer a.lk.Unlock()
	a.inst = inst
}

func (a *Arp) Run() {
	a.RunWait()
}

func (a *Arp) RunWait() {
	for {
		for i := 0; i < len(a.notes); i++ {
			if a.notes[0] <= 0 {
				time.Sleep(a.duration)
				continue
			}
			stop := a.inst.Play(a.notes[i])
			time.Sleep(a.duration)
			stop()
		}
	}
}
