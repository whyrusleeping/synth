package main

import (
	"fmt"
	"math"
	"reflect"
	"slices"
	"strconv"
	"time"
	"unicode"

	"github.com/gopxl/beep"
	"github.com/rakyll/portmidi"
)

type System struct {
	c *Stack

	vals []map[string]any
}

func (s *System) Set(k string, v any) {
	s.vals[len(s.vals)-1][k] = v
}

func NewSystem(c *Stack) *System {
	s := &System{
		c:    c,
		vals: []map[string]any{make(map[string]any)},
	}

	s.Set("makearp", MakeFunc(func(inst *Instrument, notefrac int, notes []int64) (any, error) {
		dur := time.Second / time.Duration(notefrac)
		return &Arp{
			inst:     inst,
			notes:    notes,
			duration: dur,
		}, nil
	}))

	s.Set("print", MakeFunc(func(i any) {
		fmt.Println(i)
	}))

	s.Set("getMidi", MakeFunc(func() *MidiController {
		mc, err := OpenController(portmidi.DefaultInputDeviceID())
		if err != nil {
			fmt.Println(err)
			return nil
		} else {
			return mc
		}
	}))

	s.Set("newVoice", MakeFunc(func(s beep.Streamer) *Voice {
		return &Voice{
			sub:    s,
			paused: true,
		}
	}))

	s.Set("defvoice", MakeFunc(func() func(int64) *Voice { return newVoice }))
	s.Set("defvoicef", MakeFunc(func(note int64) *Voice { return newVoice(note) }))

	s.Set("sine", MakeFunc(func(freq, amp float64) *SineWave {
		return &SineWave{sampleRate: sampleRate, amplitude: amp, frequency: freq}
	}))

	s.Set("wsine", MakeFunc(func(freq, amp float64) *WeirdSine {
		return &WeirdSine{sampleRate: sampleRate, amplitude: amp, frequency: freq}
	}))

	s.Set("ntf", MakeFunc(func(note int64) float64 {
		return 440 * math.Pow(2, (float64(note)-69)/12)
	}))

	s.Set("makeinst", MakeFunc(func(vfunc func(int64) *Voice) *Instrument {
		inst := &Instrument{
			newVoice: vfunc,
		}
		c.AddInstrument(inst)
		return inst
	}))

	s.Set("oscSine", MakeFunc(func(notefrac int, notes []int64) (any, error) {
		dur := time.Second / time.Duration(notefrac)
		return &Arp{
			notes:    notes,
			duration: dur,
		}, nil
	}))

	s.Set("sineWave", MakeFunc(func(bfreq float64) (any, error) {
		fn := func(note int64) *SineWave {
			sw := &SineWave{sampleRate: sampleRate, amplitude: 0.05}
			sw.frequency = bfreq * math.Pow(2, (float64(note)-69)/12)
			return sw
		}

		return fn, nil
	}))

	return s
}

type Function struct {
	fn reflect.Value
}

// TODO: do away with this wrapper
func MakeFunc(fn any) *Function {
	return &Function{
		fn: reflect.ValueOf(fn),
	}
}

func (f *Function) Call(args []any) (any, error) {
	return callFunc(f.fn, args)
}

func callFunc(rfv reflect.Value, args []any) (any, error) {
	t := rfv.Type()
	nargs := t.NumIn()
	var inargs []reflect.Value
	for i := 0; i < nargs; i++ {
		in := rfv.Type().In(i)

		inval, err := argToType(args[i], in)
		if err != nil {
			return nil, err
		}

		inargs = append(inargs, reflect.ValueOf(inval))
	}

	out := rfv.Call(inargs)

	if len(out) == 0 {
		return nil, nil
	}

	return out[0].Interface(), nil
}

func argToType(arg any, t reflect.Type) (any, error) {
	/*
		switch t {
		case reflect.TypeOf(time.Duration(0)):
			sval, ok := arg.(string)
			if !ok {
				return nil, fmt.Errorf("unsupported arg type: %T", arg)
			}

			return time.ParseDuration(sval)
		default:
		}
	*/

	switch t.Kind() {
	case reflect.Slice:
		arrarg, ok := arg.([]any)
		if !ok {
			return nil, fmt.Errorf("arg for array must be []any")
		}

		switch t.Elem().Kind() {
		case reflect.Int:
			var out []int
			for _, v := range arrarg {
				out = append(out, v.(int))
			}
			return out, nil
		case reflect.Int64:
			var out []int64
			for _, v := range arrarg {
				out = append(out, int64(v.(int)))
			}
			return out, nil
		default:
			return nil, fmt.Errorf("unrecognized array type: %s", t.Elem().Kind())
		}

	case reflect.Int:
		ival, ok := arg.(int)
		if !ok {
			return nil, fmt.Errorf("unsupported int arg type: %T", arg)
		}

		return ival, nil
	case reflect.Int64:
		switch arg := arg.(type) {
		case int:
			return int64(arg), nil
		case int64:
			return arg, nil
		default:
			return nil, fmt.Errorf("unsupported int64 arg type: %T", arg)
		}

	case reflect.Float64:
		switch arg := arg.(type) {
		case float64:
			return arg, nil
		case int:
			return float64(arg), nil
		default:
			return nil, fmt.Errorf("unsupported float64 arg type: %T", arg)
		}
	case reflect.Func:
		if farg, ok := arg.(*Function); ok {
			fmt.Printf("Function arg: %#v\n", farg.fn)
			return farg.fn.Interface(), nil
		}
		if dfarg, ok := arg.(*DefFunc); ok {
			fn, err := dfarg.sys.instantiateFunction(dfarg, t)
			if err != nil {
				return nil, err
			}

			return fn.fn.Interface(), nil
		}
		return arg, nil
	case reflect.Pointer:
		if reflect.TypeOf(arg).Kind() == reflect.Interface {
			// need to cast it
			//reflect.ValueOf(arg).Type().AssignableTo(t)
			return reflect.ValueOf(arg).Convert(t).Interface(), nil
		}
		return arg, nil
	case reflect.Interface:
		return arg, nil
	default:
		fmt.Printf("have type %T\n", arg)
		return nil, fmt.Errorf("requested type unknown: %s", t)
	}
}

func (s *System) ProcessCmd(cmdl string) error {
	tokens, err := tokenize(cmdl)
	if err != nil {
		return err
	}

	return s.processCmd(tokens)
}

func (s *System) processCmd(tokens []string) error {
	if len(tokens) == 0 {
		return nil
	}

	if len(tokens) == 1 {
		val, ok := s.Lookup(tokens[0])
		if !ok {
			return fmt.Errorf("unknown reference %q", tokens[0])
		}
		fmt.Println(val)
		return nil
	}

	if tokens[0] == "return" {
		val, err := s.ResolveStatement(tokens[1:])
		if err != nil {
			return err
		}

		s.Set("__RETURN_VALUE", val)
		return nil
	}

	if len(tokens) > 2 && tokens[1] == "=" {
		// assignment

		val, err := s.ResolveStatement(tokens[2:])
		if err != nil {
			return err
		}

		s.Set(tokens[0], val)
		return nil
	}

	if len(tokens) > 2 && tokens[1] == "." {
		vref, ok := s.Lookup(tokens[0])
		if !ok {
			return fmt.Errorf("unknown reference %q", tokens[0])
		}

		memname := tokens[2]
		if len(tokens) == 3 {
			return printVarMember(vref, memname)
		}

		if tokens[3] == "(" {
			args, _, err := scanTuple("(", ")", tokens[3:])
			if err != nil {
				return err
			}
			var params []any
			for i, argset := range args {
				v, err := s.ResolveStatement(argset)
				if err != nil {
					fmt.Printf("%#v\n", args)
					return fmt.Errorf("parsing arg %d: %w", i, err)
				}
				params = append(params, v)
			}

			return callMethod(vref, memname, params)
		}
	}

	vf, ok := s.Lookup(tokens[0])
	if ok {
		_, fok := vf.(*Function)
		if fok {
			_, err := s.ResolveStatement(tokens)
			if err != nil {
				return err
			}
			return nil
		}

		_, dfok := vf.(*DefFunc)
		if dfok {
			_, err := s.ResolveStatement(tokens)
			if err != nil {
				return err
			}
			return nil
		}
	}

	return fmt.Errorf("unknown command type (%#v)", tokens)
}

func callMethod(obj any, mname string, args []any) error {
	val := reflect.ValueOf(obj)
	meth := val.MethodByName(mname)

	if meth.Equal(reflect.Value{}) {
		return fmt.Errorf("%q not found", mname)
	}

	resp, err := callFunc(meth, args)
	if err != nil {
		return err
	}
	fmt.Println(resp)
	return nil
}

func printVarMember(obj any, mname string) error {
	val := reflect.ValueOf(obj)
	f := val.FieldByName(mname)
	fmt.Println(f.Interface())
	return nil
}

func (s *System) Lookup(val string) (any, bool) {
	for i := len(s.vals) - 1; i >= 0; i-- {
		ov, ok := s.vals[i][val]
		if ok {
			return ov, true
		}
	}

	return nil, false
}

func (s *System) Get(val string) any {
	ov, ok := s.Lookup(val)
	if ok {
		return ov
	}
	return nil
}

func (s *System) ResolveStatement(tokens []string) (any, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("cannot parse empty statement")
	}

	if len(tokens) == 1 {
		// either a variable or an immediate value
		val, err := strconv.Atoi(tokens[0])
		if err == nil {
			return val, nil
		}

		vbl, ok := s.Lookup(tokens[0])
		if !ok {
			return nil, fmt.Errorf("unknown reference: %q", tokens[0])
		}

		return vbl, nil
	}

	if tokens[0] == "[" {
		// array literal
		vals, _, err := scanTuple("[", "]", tokens)
		if err != nil {
			return nil, fmt.Errorf("parsing array literal: %w", err)
		}

		var out []any
		for _, v := range vals {
			arrval, err := s.ResolveStatement(v)
			if err != nil {
				return nil, err
			}
			out = append(out, arrval)
		}

		return out, nil
	}

	if tokens[0] == "func" {
		// create a new func on the fly
		return s.makeFunc(tokens)
	}

	v, ok := s.Lookup(tokens[0])
	if ok {
		f, fok := v.(Callable)
		if ok && fok {
			if tokens[1] != "(" {
				return nil, fmt.Errorf("call %q missing open paren", tokens[0])
			}

			args, _, err := scanTuple("(", ")", tokens[1:])
			if err != nil {
				return nil, fmt.Errorf("collecting args for function call: %w", err)
			}

			var params []any
			for i, argset := range args {
				v, err := s.ResolveStatement(argset)
				if err != nil {
					fmt.Printf("%#v\n", args)
					return nil, fmt.Errorf("parsing arg %d: %w", i, err)
				}
				params = append(params, v)
			}

			return f.Call(params)
		}
	}

	return nil, fmt.Errorf("invalid statement (unknown symbol %q)", tokens[0])
}

type Callable interface {
	Call(args []any) (any, error)
}

type DefFunc struct {
	sys *System

	Params []string
	Lines  [][]string
}

func (df *DefFunc) Call(args []any) (any, error) {

	var i any
	var in []reflect.Type
	for range df.Params {
		in = append(in, reflect.TypeOf(&i).Elem())
	}

	out := []reflect.Type{reflect.TypeOf(&i).Elem()}
	ft := reflect.FuncOf(in, out, false)

	fn, err := df.sys.instantiateFunction(df, ft)
	if err != nil {
		return nil, err
	}

	return fn.Call(args)
}

func (s *System) makeFunc(args []string) (*DefFunc, error) {
	/*
		func(b, c d) {
			m = 600
			return newvoice(m)
		}
	*/

	if args[1] != "(" {
		return nil, fmt.Errorf("invalid function declaration, expected '('")
	}

	names, end, err := scanTuple("(", ")", args[1:])
	if err != nil {
		return nil, err
	}

	if args[end+2] != "{" {
		return nil, fmt.Errorf("expected function open bracket")
	}

	var i any
	var in []reflect.Type
	for range names {
		in = append(in, reflect.TypeOf(&i).Elem())
	}

	lines := groupLines(args)

	if lines[len(lines)-1][0] != "}" {
		return nil, fmt.Errorf("must end functions with close bracket")
	}

	lines = lines[1 : len(lines)-1]

	var pnames []string
	for _, vals := range names {
		pnames = append(pnames, vals[0])
	}

	return &DefFunc{
		sys:    s,
		Params: pnames,
		Lines:  lines,
	}, nil
}

func (s *System) instantiateFunction(df *DefFunc, target reflect.Type) (*Function, error) {

	rfv := reflect.MakeFunc(target, func(in []reflect.Value) []reflect.Value {
		local := make(map[string]any)
		s.vals = append(s.vals, local)

		for i := 0; i < len(in); i++ {
			s.Set(df.Params[i], in[i].Interface())
		}

		defer func() {
			s.vals = s.vals[:len(s.vals)-1]
		}()

		for _, fl := range df.Lines {
			if err := s.processCmd(fl); err != nil {
				fmt.Println("SUBCALL ERROR: ", err)
				var i any
				return []reflect.Value{reflect.ValueOf(&i).Elem()}
			}
		}

		switch target.NumOut() {
		case 0:
			return []reflect.Value{}
		case 1:
			rval, ok := s.Lookup("__RETURN_VALUE")
			if ok {
				return []reflect.Value{reflect.ValueOf(rval)}
			}
			var i any
			return []reflect.Value{reflect.ValueOf(&i)}
		default:
			panic("only support one or zero return values right now")
		}

	})
	return &Function{fn: rfv}, nil
}

func groupLines(tokens []string) [][]string {
	var out [][]string
	for {
		ix := slices.Index(tokens, "\n")
		if ix == -1 {
			out = append(out, tokens)
			return out
		}

		line := tokens[:ix]
		out = append(out, line)
		tokens = tokens[ix+1:]
	}
}

// scans tokens of the form ( a(b), 123, f(d(4)))
// returns [][]string{ ["a", "(", "b", ")"], ["123"], [ "f", "(", "d", "(", "4", ")", ")" ] }
func scanTuple(beg, end string, tokens []string) ([][]string, int, error) {
	if tokens[0] != beg {
		return nil, 0, fmt.Errorf("expected %q at beginning of sequence", beg)
	}

	var out [][]string

	var cur int = 1
	var term []string
	for i := 1; i < len(tokens); i++ {
		if tokens[i] == "(" {
			term = append(term, ")")
			continue
		}
		if tokens[i] == "[" {
			term = append(term, "]")
			continue
		}

		if len(term) > 0 {
			if tokens[i] == term[len(term)-1] {
				term = term[:len(term)-1]
			}
			continue
		}

		if tokens[i] == "," {
			if i-cur == 0 {
				return nil, 0, fmt.Errorf("empty argument at index %d", len(out))
			}

			out = append(out, tokens[cur:i])
			cur = i + 1
		}

		if tokens[i] == end {
			if i > cur {
				out = append(out, tokens[cur:i])
			}
			return out, i, nil
		}
	}

	return nil, 0, fmt.Errorf("missing close sigil")
}

func tokenize(s string) ([]string, error) {
	var out []string
	var wordstart int
	inword := false
	runes := []rune(s)
	for i := 0; i < len(runes); i++ {
		switch {
		case unicode.IsLetter(runes[i]) || unicode.IsDigit(runes[i]):
			if !inword {
				inword = true
				wordstart = i
			}
		case unicode.IsSpace(runes[i]) && runes[i] != '\n':
			if inword {
				out = append(out, string(runes[wordstart:i]))
				inword = false
			}
		case runes[i] == '.',
			runes[i] == '=',
			runes[i] == '\n',
			runes[i] == ',',
			runes[i] == '(',
			runes[i] == ')',
			runes[i] == '[',
			runes[i] == ']',
			runes[i] == '{',
			runes[i] == '}':
			if inword {
				out = append(out, string(runes[wordstart:i]))
				inword = false
			}
			out = append(out, string(runes[i]))
		default:
			return nil, fmt.Errorf("invalid character at index %d: %q", i, runes[i])
		}
	}
	if inword {
		out = append(out, string(runes[wordstart:]))

	}

	return out, nil
}
