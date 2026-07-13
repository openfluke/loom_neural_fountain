package nfountain

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/openfluke/loom/poly"
)

// SpectrumConfig controls the full Numerical×Layer fountain showcase.
type SpectrumConfig struct {
	Quick      bool
	LogPath    string
	K          int
	Epochs     int
	LossRate   float64
	UseExact   bool
	Families   []string // empty → all
	DTypes     []poly.DType // empty → all 21
}

func DefaultSpectrumConfig() SpectrumConfig {
	return SpectrumConfig{
		K:        3,
		Epochs:   1,
		LossRate: 0.25,
		UseExact: true,
		LogPath:  "",
	}
}

type spectrumCase struct {
	Family string
	DType  poly.DType
}

type spectrumResult struct {
	Family       string
	DType        string
	OK           bool
	Err          string
	BlobBytes    int
	Received     int
	Sprayed      int
	Recovered    int
	K            int
	SpecializeUs int64
	FountainUs   int64
	TotalUs      int64
	ForwardOK    bool
	Exact        bool
}

// RunSpectrumShowcase exercises Neural Fountain across layer families × numerical types,
// writes a detailed log with timings, and prints a summary table.
func RunSpectrumShowcase(cfg SpectrumConfig) bool {
	if cfg.K < 2 {
		cfg.K = 2
	}
	if cfg.Epochs < 1 {
		cfg.Epochs = 1
	}
	logPath := cfg.LogPath
	if logPath == "" {
		_ = os.MkdirAll("logs", 0o755)
		logPath = filepath.Join("logs", fmt.Sprintf("neural_fountain_spectrum_%s.log",
			time.Now().Format("20060102_150405")))
	} else {
		if dir := filepath.Dir(logPath); dir != "" && dir != "." {
			_ = os.MkdirAll(dir, 0o755)
		}
	}
	logFile, err := os.Create(logPath)
	if err != nil {
		fmt.Printf("  FAIL create log: %v\n", err)
		return false
	}
	defer logFile.Close()
	w := io.MultiWriter(os.Stdout, logFile)

	logf := func(format string, args ...any) {
		fmt.Fprintf(w, format+"\n", args...)
	}

	logf("╔══════════════════════════════════════════════════════════════╗")
	logf("║  Neural Fountain SPECTRUM — layers × dtypes · full log       ║")
	logf("╚══════════════════════════════════════════════════════════════╝")
	logf("started: %s", time.Now().Format(time.RFC3339))
	logf("log: %s", logPath)
	logf("config: K=%d epochs=%d loss=%.2f exact=%v quick=%v",
		cfg.K, cfg.Epochs, cfg.LossRate, cfg.UseExact, cfg.Quick)

	families := spectrumFamilies()
	dtypes := poly.SeedDTypesAll()
	if len(cfg.Families) > 0 {
		want := map[string]bool{}
		for _, f := range cfg.Families {
			want[strings.ToLower(f)] = true
		}
		filtered := families[:0]
		for _, f := range families {
			if want[f.name] {
				filtered = append(filtered, f)
			}
		}
		families = filtered
	}
	if len(cfg.DTypes) > 0 {
		dtypes = cfg.DTypes
	}
	if cfg.Quick {
		families = quickFamilies(families)
		dtypes = quickDTypes()
		logf("QUICK subset: %d families × %d dtypes", len(families), len(dtypes))
	}

	var cases []spectrumCase
	for _, f := range families {
		for _, dt := range dtypes {
			cases = append(cases, spectrumCase{Family: f.name, DType: dt})
		}
	}
	logf("total cases: %d (%d families × %d dtypes)", len(cases), len(families), len(dtypes))
	logf("")

	results := make([]spectrumResult, 0, len(cases))
	suiteStart := time.Now()
	pass, fail := 0, 0

	for i, c := range cases {
		logf("── case %d/%d  family=%s  dtype=%s ──", i+1, len(cases), c.Family, c.DType.String())
		res := runSpectrumCase(c, cfg, logf)
		results = append(results, res)
		if res.OK {
			pass++
			logf("  PASS  total=%.2fms specialize=%.2fms fountain=%.2fms blob=%d recv=%d sprayed=%d forward=%v",
				usToMs(res.TotalUs), usToMs(res.SpecializeUs), usToMs(res.FountainUs),
				res.BlobBytes, res.Received, res.Sprayed, res.ForwardOK)
		} else {
			fail++
			logf("  FAIL  total=%.2fms err=%s", usToMs(res.TotalUs), res.Err)
		}
		logf("")
	}

	elapsed := time.Since(suiteStart)
	logf("══════════════════════════════════════════════════════════════")
	logf("SUMMARY")
	logf("  pass=%d  fail=%d  total=%d  wall=%s", pass, fail, len(cases), elapsed.Round(time.Millisecond))
	logf("")
	logf("%-12s %-12s %-6s %10s %10s %10s %8s %s",
		"FAMILY", "DTYPE", "OK", "TOTAL_ms", "SPEC_ms", "FNTN_ms", "BLOB", "NOTE")
	for _, r := range results {
		ok := "PASS"
		note := ""
		if !r.OK {
			ok = "FAIL"
			note = r.Err
			if len(note) > 60 {
				note = note[:57] + "..."
			}
		} else if !r.ForwardOK {
			note = "forward soft-fail"
		}
		logf("%-12s %-12s %-6s %10.2f %10.2f %10.2f %8d %s",
			r.Family, r.DType, ok, usToMs(r.TotalUs), usToMs(r.SpecializeUs), usToMs(r.FountainUs), r.BlobBytes, note)
	}
	logf("")
	logf("finished: %s", time.Now().Format(time.RFC3339))
	logf("full log written to %s", logPath)

	// Also write a compact CSV next to the log.
	csvPath := strings.TrimSuffix(logPath, filepath.Ext(logPath)) + ".csv"
	if err := writeSpectrumCSV(csvPath, results); err != nil {
		logf("csv write warning: %v", err)
	} else {
		logf("csv written to %s", csvPath)
	}

	return fail == 0
}

func runSpectrumCase(c spectrumCase, cfg SpectrumConfig, logf func(string, ...any)) (res spectrumResult) {
	res = spectrumResult{
		Family: c.Family,
		DType:  c.DType.String(),
		Exact:  cfg.UseExact,
		K:      cfg.K,
	}
	start := time.Now()
	defer func() { res.TotalUs = time.Since(start).Microseconds() }()

	factory, batches, err := spectrumFactoryAndBatches(c.Family, c.DType, cfg.K)
	if err != nil {
		res.Err = "setup: " + err.Error()
		return
	}

	pcfg := poly.DefaultNeuralFountainConfig()
	pcfg.K = cfg.K
	pcfg.Epochs = cfg.Epochs
	pcfg.LossRate = cfg.LossRate
	pcfg.UseExactDType = cfg.UseExact
	pcfg.UniformDType = c.DType
	pcfg.Verbose = false
	pcfg.MaxOverhead = 8.0
	pcfg.Seed = poly.SeedFrom("nf-spectrum", c.Family, c.DType.String())

	master, err := poly.NeuralFountain(factory, batches, pcfg)
	if err != nil {
		res.Err = err.Error()
		return
	}
	res.OK = true
	res.Recovered = master.Recovered
	res.Received = master.Received
	res.Sprayed = master.Sprayed
	res.SpecializeUs = master.SpecializeUs
	res.FountainUs = master.FountainUs
	if len(master.Experts) > 0 && master.Experts[0] != nil {
		if blob, err := poly.PackNetworkWeights(master.Experts[0]); err == nil {
			res.BlobBytes = len(blob)
		}
	}

	// Forward smoke on first batch input.
	if len(batches) > 0 && batches[0].Input != nil {
		if _, err := master.Forward(batches[0].Input); err == nil {
			res.ForwardOK = true
		} else {
			logf("  forward warn: %v", err)
		}
	}
	return
}

func writeSpectrumCSV(path string, results []spectrumResult) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	fmt.Fprintln(f, "family,dtype,ok,err,blob_bytes,received,sprayed,recovered,k,total_us,specialize_us,fountain_us,forward_ok,exact")
	for _, r := range results {
		errEsc := strings.ReplaceAll(r.Err, `"`, `'`)
		fmt.Fprintf(f, "%s,%s,%t,\"%s\",%d,%d,%d,%d,%d,%d,%d,%d,%t,%t\n",
			r.Family, r.DType, r.OK, errEsc, r.BlobBytes, r.Received, r.Sprayed, r.Recovered, r.K,
			r.TotalUs, r.SpecializeUs, r.FountainUs, r.ForwardOK, r.Exact)
	}
	return nil
}

func usToMs(us int64) float64 {
	return float64(us) / 1000.0
}

func quickDTypes() []poly.DType {
	return []poly.DType{
		poly.DTypeFloat32,
		poly.DTypeFloat16,
		poly.DTypeBFloat16,
		poly.DTypeInt8,
		poly.DTypeInt4,
		poly.DTypeTernary,
		poly.DTypeBinary,
	}
}

func quickFamilies(all []layerFamily) []layerFamily {
	// All families; quick only shrinks the dtype axis.
	return all
}
