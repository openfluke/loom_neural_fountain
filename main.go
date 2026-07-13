package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/openfluke/chaosglue/loom_neural_fountain/nfountain"
	"github.com/openfluke/loom/poly"
)

func main() {
	args := os.Args[1:]
	mode := "mnist"
	if len(args) > 0 {
		switch strings.ToLower(args[0]) {
		case "showcase", "spectrum":
			mode = "showcase"
			args = args[1:]
		}
	}

	quick := false
	dataDir := "data"
	if _, err := os.Stat("../loom_seed_mnist/data/train-images-idx3-ubyte.gz"); err == nil {
		dataDir = "../loom_seed_mnist/data"
	}
	if _, err := os.Stat("../loom_fountain_codes/data/train-images-idx3-ubyte.gz"); err == nil && dataDir == "data" {
		dataDir = "../loom_fountain_codes/data"
	}

	mnist := nfountain.DefaultConfig()
	spec := nfountain.DefaultSpectrumConfig()

	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "-showcase" || a == "--showcase" || a == "-spectrum" || a == "--spectrum":
			mode = "showcase"
		case a == "-quick" || a == "--quick" || a == "quick":
			quick = true
		case a == "-k" && i+1 < len(args):
			i++
			if v, err := strconv.Atoi(args[i]); err == nil && v > 1 {
				mnist.K = v
				spec.K = v
			}
		case a == "-epochs" && i+1 < len(args):
			i++
			if v, err := strconv.Atoi(args[i]); err == nil && v > 0 {
				mnist.SpecialistEP = v
				spec.Epochs = v
			}
		case a == "-loss" && i+1 < len(args):
			i++
			if v, err := strconv.ParseFloat(args[i], 64); err == nil {
				mnist.LossRate = v
				spec.LossRate = v
			}
		case a == "-log" && i+1 < len(args):
			i++
			spec.LogPath = args[i]
		case a == "-family" && i+1 < len(args):
			i++
			spec.Families = splitCSV(args[i])
		case a == "-dtype" && i+1 < len(args):
			i++
			dts, err := parseDTypes(args[i])
			if err != nil {
				fmt.Fprintf(os.Stderr, "%v\n", err)
				os.Exit(2)
			}
			spec.DTypes = dts
		case a == "-strict" || a == "--strict":
			spec.Strict = true
		case a == "-exact" || a == "--exact":
			spec.UseExact = true
		case a == "-no-exact" || a == "--no-exact":
			spec.UseExact = false
		case a == "-min-oracle" && i+1 < len(args):
			i++
			if v, err := strconv.ParseFloat(args[i], 64); err == nil {
				spec.MinOracle = v
			}
		case a == "-batches" && i+1 < len(args):
			i++
			if v, err := strconv.Atoi(args[i]); err == nil && v > 0 {
				spec.Batches = v
			}
		case a == "-h" || a == "--help" || a == "help":
			printUsage()
			return
		case a[0] != '-':
			if mode == "mnist" {
				dataDir = a
			} else {
				fmt.Fprintf(os.Stderr, "unknown arg %s\n", a)
				os.Exit(2)
			}
		default:
			fmt.Fprintf(os.Stderr, "unknown flag %s\n", a)
			printUsage()
			os.Exit(2)
		}
	}

	if mode == "showcase" {
		spec.Quick = quick
		if !nfountain.RunSpectrumShowcase(spec) {
			os.Exit(1)
		}
		return
	}

	if quick {
		_ = os.Setenv("LOOM_NEURAL_FOUNTAIN_QUICK", "1")
		mnist = nfountain.DefaultConfig()
		mnist.Quick = true
		mnist.K = nfountain.DefaultConfig().K
		mnist.SpecialistEP = nfountain.DefaultConfig().SpecialistEP
		mnist.HeatVal = nfountain.DefaultConfig().HeatVal
		// re-apply any -k/-epochs/-loss that came after quick reset
		for i := 0; i < len(args); i++ {
			a := args[i]
			switch {
			case a == "-k" && i+1 < len(args):
				i++
				if v, err := strconv.Atoi(args[i]); err == nil && v > 1 {
					mnist.K = v
				}
			case a == "-epochs" && i+1 < len(args):
				i++
				if v, err := strconv.Atoi(args[i]); err == nil && v > 0 {
					mnist.SpecialistEP = v
				}
			case a == "-loss" && i+1 < len(args):
				i++
				if v, err := strconv.ParseFloat(args[i], 64); err == nil {
					mnist.LossRate = v
				}
			}
		}
	}

	if dataDir != "data" {
		fmt.Printf("using MNIST cache from %s\n", dataDir)
	}
	_ = filepath.IsAbs(dataDir)
	if !nfountain.Run(dataDir, mnist) {
		os.Exit(1)
	}
}

func splitCSV(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(strings.ToLower(p))
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func parseDTypes(s string) ([]poly.DType, error) {
	names := splitCSV(s)
	out := make([]poly.DType, 0, len(names))
	for _, name := range names {
		dt := poly.ParseDType(name)
		matched := false
		for _, known := range poly.SeedDTypesAll() {
			if known == dt || strings.EqualFold(known.String(), name) {
				out = append(out, known)
				matched = true
				break
			}
		}
		if !matched {
			return nil, fmt.Errorf("unknown dtype %q", name)
		}
	}
	return out, nil
}

func printUsage() {
	fmt.Fprintf(os.Stderr, `loom_neural_fountain — Neural Fountain demos

Usage:
  go run . [flags] [dataDir]           MNIST Master assemble (default)
  go run . showcase [flags]            layer × dtype fountain spectrum
  go run . -showcase [flags]

MNIST flags:
  -quick          4k train subset · K=8
  -k N            specialists / shards
  -epochs N       train epochs per specialist
  -loss R         fountain erase rate

Showcase flags:
  -quick                  subset of dtypes (smoke)
  -k N                    specialists (default 8; quick→4)
  -epochs N               epochs per specialist (default 40; quick→12)
  -batches N              micro-task samples (default K*16)
  -loss R                 fountain erase rate (default 0.25)
  -min-oracle PCT         quality target (dense oracle ≥90; other families fit ≥55)
  -strict                 treat quality WEAK as run failure (default: warn only)
  -log PATH               write full log here (default logs/neural_fountain_spectrum_*.log)
  -family a,b,c           limit families (dense,swiglu,mha,cnn1,cnn2,cnn3,rnn,lstm,embedding,residual)
  -dtype a,b              limit dtypes (e.g. float32,float16,int8)
  -exact / -no-exact      native-dtype train stress (default off = FP32 SoT + morph)

Pipeline (layers mode):
  micro-specialize shards → pack FP32 Masters → LT spray/peel → unpack Master
  → score oracle/ensemble → morph experts to case dtype
  Dense classification targets ≥90% oracle (micro-shards consolidating through fountain).


Examples:
  ./run.sh
  ./run.sh quick
  ./run.sh showcase
  ./run.sh showcase quick
  ./run.sh showcase -family dense,cnn1 -dtype float32,int8
`)
}
