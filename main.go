package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"github.com/openfluke/chaosglue/loom_neural_fountain/nfountain"
)

func main() {
	cfg := nfountain.DefaultConfig()
	dataDir := "data"
	if _, err := os.Stat("../loom_seed_mnist/data/train-images-idx3-ubyte.gz"); err == nil {
		dataDir = "../loom_seed_mnist/data"
		fmt.Println("using MNIST cache from ../loom_seed_mnist/data")
	}
	if _, err := os.Stat("../loom_fountain_codes/data/train-images-idx3-ubyte.gz"); err == nil && dataDir == "data" {
		dataDir = "../loom_fountain_codes/data"
	}

	for i := 1; i < len(os.Args); i++ {
		a := os.Args[i]
		switch {
		case a == "-k" && i+1 < len(os.Args):
			i++
			if v, err := strconv.Atoi(os.Args[i]); err == nil && v > 1 {
				cfg.K = v
			}
		case a == "-epochs" && i+1 < len(os.Args):
			i++
			if v, err := strconv.Atoi(os.Args[i]); err == nil && v > 0 {
				cfg.SpecialistEP = v
			}
		case a == "-loss" && i+1 < len(os.Args):
			i++
			if v, err := strconv.ParseFloat(os.Args[i], 64); err == nil {
				cfg.LossRate = v
			}
		case a == "-quick" || a == "--quick":
			cfg.Quick = true
		case a[0] != '-':
			dataDir = a
		default:
			fmt.Fprintf(os.Stderr, "unknown flag %s\n", a)
			os.Exit(2)
		}
	}

	if cfg.Quick {
		_ = os.Setenv("LOOM_NEURAL_FOUNTAIN_QUICK", "1")
		cfg = nfountain.DefaultConfig()
		cfg.Quick = true
	}

	root, _ := os.Getwd()
	_ = root
	if !filepath.IsAbs(dataDir) {
		// keep relative
	}
	if !nfountain.Run(dataDir, cfg) {
		os.Exit(1)
	}
}
