package main

import (
	"encoding/json"
	"fmt"

	"github.com/rakyll/portmidi"
)

type MidiController struct {
	Target *Instrument

	stream *portmidi.Stream

	noteStates map[int64]func()

	knobsSeen map[int64]*knobInfo

	knobBinds map[int64]*knobBind
}

type knobBind struct {
	mapf func(int64) float64
	sf   Setter
}

func (kb *knobBind) Update(val int64) {
	v := kb.mapf(val)
	kb.sf(v)
}

type knobInfo struct {
	lastVal int64
}

func OpenController(id portmidi.DeviceID) (*MidiController, error) {
	in, err := portmidi.NewInputStream(id, 1024)
	if err != nil {
		return nil, err
	}

	mc := &MidiController{
		stream:     in,
		noteStates: make(map[int64]func()),
		knobsSeen:  make(map[int64]*knobInfo),
		knobBinds:  make(map[int64]*knobBind),
	}

	go mc.run()

	return mc, nil
}

func NewMockController() *MidiController {
	mc := &MidiController{
		noteStates: make(map[int64]func()),
		knobsSeen:  make(map[int64]*knobInfo),
		knobBinds:  make(map[int64]*knobBind),
	}

	return mc
}

func (mc *MidiController) Shutdown() {
	mc.stream.Close()
}

func (mc *MidiController) SetInst(inst *Instrument) {
	mc.Target = inst
}

func (mc *MidiController) run() {
	for {
		events, err := mc.stream.Read(1024)
		if err != nil {
			panic(err)
		}

		for _, event := range events {
			b, err := json.Marshal(event)
			if err != nil {
				panic(err)
			}
			fmt.Println(string(b))
			switch event.Status {
			case 0x90:
				note := int64(event.Data1)
				mc.startNote(note)
			case 0x80:
				note := int64(event.Data1)
				mc.stopNote(note)
			case 0xb0:
				// twisty knobs

				ki, ok := mc.knobsSeen[event.Data1]
				if !ok {
					ki = &knobInfo{}
					mc.knobsSeen[event.Data1] = ki
				}
				ki.lastVal = event.Data2

				kb, ok := mc.knobBinds[event.Data1]
				if ok {
					kb.Update(event.Data2)
				}

				//controller.filter.UpdateCutoff(10000 * (float64(event.Data2) / 128))
				//controller.filter.UpdateCutoff(math.Pow(float64(event.Data2), 1.5))

			default:
				b, err := json.Marshal(event)
				if err != nil {
					panic(err)
				}
				fmt.Println(string(b))
			}
		}
	}
}

func (mc *MidiController) startNote(note int64) {
	if mc.Target == nil {
		return
	}

	if oldstop, ok := mc.noteStates[note]; ok {
		fmt.Println("got start for already running note")
		oldstop()
	}
	stop := mc.Target.Play(note)
	mc.noteStates[note] = stop
}

func (mc *MidiController) stopNote(note int64) {
	stopf, ok := mc.noteStates[note]
	if !ok {
		fmt.Println("stop called on note we hadnt started")
		return
	}

	stopf()
	delete(mc.noteStates, note)
}

type Settable interface {
	GetSetter(string) func(float64)
}

type Setter func(float64)

func (mc *MidiController) BindKnob(knobid int64, s Setter, rangeMapFunc func(int64) float64) {
	if s == nil {
		fmt.Println("nil setter passed to bind knob: ", knobid)
		return
	}
	mc.knobBinds[knobid] = &knobBind{
		mapf: rangeMapFunc,
		sf:   s,
	}
}
