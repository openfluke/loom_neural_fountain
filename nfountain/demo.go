package nfountain

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/openfluke/loom/neural"
	"github.com/openfluke/loom/poly"
)

const mnistPixels = 28 * 28

type Config struct {
	K            int
	SpecialistEP int
	LossRate     float64
	Quick        bool
	HeatVal      int
}

func DefaultConfig() Config {
	cfg := Config{
		K:            16,
		SpecialistEP: 5,
		LossRate:     0.30,
		HeatVal:      1500,
	}
	if os.Getenv("LOOM_NEURAL_FOUNTAIN_QUICK") == "1" {
		cfg.Quick = true
		cfg.K = 8
		cfg.SpecialistEP = 3
		cfg.HeatVal = 400
	}
	return cfg
}

func Run(dataDir string, cfg Config) bool {
	fmt.Println("╔══════════════════════════════════════════════════════════╗")
	fmt.Println("║  Neural Fountain — specialists · LT peel · Master net    ║")
	fmt.Println("╚══════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("Paradigm (loom/neural): same spray/peel that recovers 56k pixels,")
	fmt.Println("but source blocks = trained specialist WEIGHTS (poly.Train, no seed hunt).")
	fmt.Println("Recover all specialists byte-exact → assemble Master ensemble.")
	fmt.Println("Oracle routing on train ≈ covering every train sample (the 100% story).")
	fmt.Println()

	fmt.Println("── MNIST data ──")
	train, val, err := loadMNIST8020(dataDir)
	if err != nil {
		fmt.Printf("  FAIL: %v\n", err)
		return false
	}
	fmt.Printf("  train=%d  val=%d\n", len(train), len(val))

	if cfg.Quick && len(train) > 4096 {
		train = train[:4096]
		fmt.Printf("  QUICK: using train subset n=%d\n", len(train))
	}

	acfg := neural.DefaultAssembleConfig()
	acfg.K = cfg.K
	acfg.SpecialistEP = cfg.SpecialistEP
	acfg.LossRate = cfg.LossRate
	acfg.Sizes = []int{mnistPixels, 128, 64, 10}
	acfg.Verbose = true

	master, err := neural.AssembleMaster(train, acfg)
	if err != nil {
		fmt.Printf("  FAIL assemble: %v\n", err)
		return false
	}

	oracle := neural.EvalOracleTrainAccuracy(master, train)
	ensTrain := neural.EvalEnsembleAccuracy(master, train)
	ensVal := neural.EvalEnsembleAccuracy(master, valSubset(val, cfg.HeatVal))
	meanVal := neural.EvalMeanNetAccuracy(master, valSubset(val, cfg.HeatVal))

	fmt.Println("\n── Master scores ──")
	fmt.Printf("  oracle train coverage   = %.2f%%  (shard expert owns each train point)\n", oracle*100)
	fmt.Printf("  ensemble train          = %.2f%%\n", ensTrain*100)
	fmt.Printf("  ensemble val            = %.2f%%\n", ensVal*100)
	if master.MeanNet != nil {
		fmt.Printf("  weight-mean net val     = %.2f%%\n", meanVal*100)
	}
	fmt.Printf("  recovered specialists   = %d/%d (fountain byte-exact)\n", master.Recovered, master.K)

	printHeatmap(master, valSubset(val, cfg.HeatVal), "val ensemble AFTER Neural Fountain")

	if oracle < 0.85 {
		fmt.Println("\n  note: oracle train < 85% — specialists underfit shards; raise -epochs")
	} else {
		fmt.Println("\n✓ Oracle coverage high — every train shard’s knowledge recovered through the fountain.")
	}
	fmt.Println("✓ Master assembled without layer_seed search.")
	return true
}

func valSubset(val []neural.Sample, n int) []neural.Sample {
	if n <= 0 || n >= len(val) {
		return val
	}
	return val[:n]
}

func printHeatmap(m *neural.Master, set []neural.Sample, title string) {
	fmt.Printf("\n── %s · n=%d ──\n", title, len(set))
	dm := poly.NewDeviationMetrics()
	for i, s := range set {
		pred, err := m.PredictArgmax(s.X)
		actual := -1.0
		if err == nil {
			actual = float64(pred)
		}
		dm.UpdateMetrics(poly.EvaluatePrediction(i, float64(s.Y), actual))
	}
	dm.ComputeFinalMetrics()
	dm.PrintSummary()
}

var mnistFiles = []struct {
	name string
	urls []string
}{
	{"train-images-idx3-ubyte.gz", []string{
		"https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
		"https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
	}},
	{"train-labels-idx1-ubyte.gz", []string{
		"https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
		"https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
	}},
	{"t10k-images-idx3-ubyte.gz", []string{
		"https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
		"https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
	}},
	{"t10k-labels-idx1-ubyte.gz", []string{
		"https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
		"https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
	}},
}

func loadMNIST8020(dataDir string) (train, val []neural.Sample, err error) {
	if err := ensureMNIST(dataDir); err != nil {
		return nil, nil, err
	}
	trX, err := loadImages(filepath.Join(dataDir, "train-images-idx3-ubyte.gz"))
	if err != nil {
		return nil, nil, err
	}
	trY, err := loadLabels(filepath.Join(dataDir, "train-labels-idx1-ubyte.gz"))
	if err != nil {
		return nil, nil, err
	}
	teX, err := loadImages(filepath.Join(dataDir, "t10k-images-idx3-ubyte.gz"))
	if err != nil {
		return nil, nil, err
	}
	teY, err := loadLabels(filepath.Join(dataDir, "t10k-labels-idx1-ubyte.gz"))
	if err != nil {
		return nil, nil, err
	}

	all := make([]neural.Sample, 0, len(trX)+len(teX))
	for i := range trX {
		all = append(all, neural.Sample{X: bytesToF32(trX[i]), Y: trY[i]})
	}
	for i := range teX {
		all = append(all, neural.Sample{X: bytesToF32(teX[i]), Y: teY[i]})
	}
	rng := poly.NewSeedRNG(poly.SeedFrom("loom-neural-fountain-shuffle", uint64(len(all))))
	for i := len(all) - 1; i > 0; i-- {
		j := int(rng.Uint64() % uint64(i+1))
		all[i], all[j] = all[j], all[i]
	}
	nTrain := int(0.8 * float64(len(all)))
	return all[:nTrain], all[nTrain:], nil
}

func bytesToF32(b []byte) []float32 {
	out := make([]float32, mnistPixels)
	for i := 0; i < mnistPixels && i < len(b); i++ {
		out[i] = float32(b[i]) / 255.0
	}
	return out
}

func ensureMNIST(dataDir string) error {
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return err
	}
	for _, f := range mnistFiles {
		path := filepath.Join(dataDir, f.name)
		if st, err := os.Stat(path); err == nil && st.Size() > 0 {
			fmt.Printf("  cached %s (%d bytes)\n", f.name, st.Size())
			continue
		}
		fmt.Printf("  downloading %s …\n", f.name)
		var last error
		for _, url := range f.urls {
			if err := downloadFile(url, path); err != nil {
				last = err
				continue
			}
			last = nil
			break
		}
		if last != nil {
			return fmt.Errorf("download %s: %w", f.name, last)
		}
	}
	return nil
}

func downloadFile(url, dest string) error {
	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	tmp := dest + ".partial"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(out, resp.Body)
	closeErr := out.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	return os.Rename(tmp, dest)
}

func loadImages(path string) ([][]byte, error) {
	raw, err := readGzip(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 16 || binary.BigEndian.Uint32(raw[0:4]) != 2051 {
		return nil, fmt.Errorf("bad image magic")
	}
	n := int(binary.BigEndian.Uint32(raw[4:8]))
	out := make([][]byte, n)
	off := 16
	for i := 0; i < n; i++ {
		pix := make([]byte, mnistPixels)
		copy(pix, raw[off:off+mnistPixels])
		out[i] = pix
		off += mnistPixels
	}
	return out, nil
}

func loadLabels(path string) ([]int, error) {
	raw, err := readGzip(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 8 || binary.BigEndian.Uint32(raw[0:4]) != 2049 {
		return nil, fmt.Errorf("bad label magic")
	}
	n := int(binary.BigEndian.Uint32(raw[4:8]))
	out := make([]int, n)
	for i := 0; i < n; i++ {
		out[i] = int(raw[8+i])
	}
	return out, nil
}

func readGzip(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer zr.Close()
	return io.ReadAll(zr)
}
