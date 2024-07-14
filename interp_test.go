package main

import (
	"testing"
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
