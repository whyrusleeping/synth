package main

import (
	"fmt"
	"testing"
)

func TestTokenize(t *testing.T) {
	cmd := "foo = a.bar(500ms ,beep, bop )"
	tokens, err := tokenize(cmd)
	if err != nil {
		t.Fatal(err)
	}

	for _, t := range tokens {
		fmt.Println(t)
	}
}
