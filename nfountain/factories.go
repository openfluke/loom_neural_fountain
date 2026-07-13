package nfountain

import (
	"fmt"
	"math"

	"github.com/openfluke/loom/poly"
)

type layerFamily struct {
	name string
}

func spectrumFamilies() []layerFamily {
	return []layerFamily{
		{name: "dense"},
		{name: "swiglu"},
		{name: "mha"},
		{name: "cnn1"},
		{name: "cnn2"},
		{name: "cnn3"},
		{name: "rnn"},
		{name: "lstm"},
		{name: "embedding"},
		{name: "residual"},
	}
}

// spectrumTask describes how batches were built so eval can score consolidate quality.
type spectrumTask struct {
	Factory  poly.NetworkFactory
	Batches  []poly.TrainingBatch[float32]
	Classify bool // argmax accuracy; else MSE fit score
	LossType string
}

func spectrumFactoryAndBatches(family string, dt poly.DType, k, nBatches int) (spectrumTask, error) {
	dtName := dt.String()
	if nBatches < k*8 {
		nBatches = k * 8
	}

	switch family {
	case "dense":
		// Micro multi-class: each shard specializes on its slice; Master consolidates.
		sizes := []int{16, 32, 16, 4}
		factory := func(idx int) (*poly.VolumetricNetwork, error) {
			topo := poly.DenseTopologySeed("nf-spectrum-dense", sizes)
			topo ^= uint64(idx+1) * 0x9e3779b97f4a7c15
			dtypes := repeatDtype(dtName, len(sizes)-1)
			m, err := poly.BuildDenseManifest(topo, sizes, dtypes)
			if err != nil {
				return nil, err
			}
			net, err := poly.BuildDenseVolumetricFromManifest(m)
			if err != nil {
				return nil, err
			}
			last := net.GetLayer(0, 0, 0, len(m.Layers)-1)
			last.Activation = poly.ActivationLinear
			return net, nil
		}
		return spectrumTask{
			Factory:  factory,
			Batches:  microClassBatches(nBatches, sizes[0], sizes[len(sizes)-1]),
			Classify: true,
			LossType: "mse",
		}, nil

	case "swiglu":
		specs := []poly.SwiGLUSpec{{Hidden: 8, Intermediate: 16}}
		factory := func(idx int) (*poly.VolumetricNetwork, error) {
			topo := poly.SwiGLUTopologySeed("nf-spectrum-swiglu", specs)
			topo ^= uint64(idx + 1)
			m, err := poly.BuildSwiGLUManifest(topo, specs, []string{dtName})
			if err != nil {
				return nil, err
			}
			return poly.BuildSwiGLUVolumetricFromManifest(m)
		}
		// Same pattern as seed_showcase: one probe input, fixed target, many repeats —
		// specialists memorize the map; fountain must still recover weights to consolidate.
		batches, err := repeatedProbeFitBatches(factory, vec(8), nBatches)
		return spectrumTask{Factory: factory, Batches: batches, LossType: "mse"}, err

	case "mha":
		specs := []poly.MHASpec{{DModel: 8, NumHeads: 2, NumKVHeads: 2, HeadDim: 4, QueryDim: 8}}
		factory := func(idx int) (*poly.VolumetricNetwork, error) {
			topo := poly.MHATopologySeed("nf-spectrum-mha", specs)
			topo ^= uint64(idx + 1)
			m, err := poly.BuildMHAManifest(topo, specs, []string{dtName})
			if err != nil {
				return nil, err
			}
			return poly.BuildMHAVolumetricFromManifest(m)
		}
		batches, err := variedVectorFitBatches(factory, 8, nBatches)
		return spectrumTask{Factory: factory, Batches: batches, LossType: "mse"}, err

	case "cnn1":
		return cnnFitTask(1, 8, dtName, nBatches)
	case "cnn2":
		return cnnFitTask(2, 8, dtName, nBatches)
	case "cnn3":
		return cnnFitTask(3, 4, dtName, nBatches)

	case "rnn":
		sizes := []int{4, 6}
		factory := func(idx int) (*poly.VolumetricNetwork, error) {
			topo := poly.RNNTopologySeed("nf-spectrum-rnn", sizes)
			topo ^= uint64(idx + 1)
			m, err := poly.BuildRNNManifest(topo, sizes, []string{dtName})
			if err != nil {
				return nil, err
			}
			return poly.BuildRNNVolumetricFromManifest(m)
		}
		batches, err := variedVectorFitBatches(factory, 4, nBatches)
		return spectrumTask{Factory: factory, Batches: batches, LossType: "mse"}, err

	case "lstm":
		sizes := []int{4, 6}
		factory := func(idx int) (*poly.VolumetricNetwork, error) {
			topo := poly.LSTMTopologySeed("nf-spectrum-lstm", sizes)
			topo ^= uint64(idx + 1)
			m, err := poly.BuildLSTMManifest(topo, sizes, []string{dtName})
			if err != nil {
				return nil, err
			}
			return poly.BuildLSTMVolumetricFromManifest(m)
		}
		batches, err := variedVectorFitBatches(factory, 4, nBatches)
		return spectrumTask{Factory: factory, Batches: batches, LossType: "mse"}, err

	case "embedding":
		spec := poly.EmbeddingSpec{VocabSize: 16, EmbeddingDim: 8, SeqLen: 4}
		factory := func(idx int) (*poly.VolumetricNetwork, error) {
			topo := poly.EmbeddingTopologySeed("nf-spectrum-emb", []poly.EmbeddingSpec{spec})
			topo ^= uint64(idx + 1)
			m, err := poly.BuildEmbeddingManifest(topo, []poly.EmbeddingSpec{spec}, []string{dtName})
			if err != nil {
				return nil, err
			}
			return poly.BuildEmbeddingVolumetricFromManifest(m)
		}
		outDim := spec.SeqLen * spec.EmbeddingDim
		batches := make([]poly.TrainingBatch[float32], nBatches)
		for i := range batches {
			tok := poly.EmbeddingDemoTokens(spec.VocabSize, spec.SeqLen)
			for j := range tok.Data {
				tok.Data[j] = float32((i*3 + j) % spec.VocabSize)
			}
			tgt := make([]float32, outDim)
			for j := range tgt {
				// Embedding should map token id → smooth value; tile by seq.
				tokID := tok.Data[j%len(tok.Data)]
				tgt[j] = tokID / float32(spec.VocabSize)
			}
			batches[i] = poly.TrainingBatch[float32]{
				Input:  tok,
				Target: poly.NewTensorFromSlice(tgt, 1, outDim),
			}
		}
		return spectrumTask{Factory: factory, Batches: batches, LossType: "mse"}, nil

	case "residual":
		spec := poly.ResidualSpec{In: 8, Out: 8}
		factory := func(idx int) (*poly.VolumetricNetwork, error) {
			topo := poly.ResidualTopologySeed("nf-spectrum-res", spec)
			topo ^= uint64(idx + 1)
			m, err := poly.BuildResidualManifest(topo, spec, dtName)
			if err != nil {
				return nil, err
			}
			return poly.BuildResidualVolumetricFromManifest(m)
		}
		batches, err := variedVectorFitBatches(factory, 8, nBatches)
		return spectrumTask{Factory: factory, Batches: batches, LossType: "mse"}, err
	}

	return spectrumTask{}, fmt.Errorf("unknown family %q", family)
}

func cnnFitTask(dim, spatial int, dtName string, nBatches int) (spectrumTask, error) {
	spec := poly.CNNSpec{Dim: dim, InputChannels: 2, Filters: 4, Spatial: spatial, KernelSize: 3}
	factory := func(idx int) (*poly.VolumetricNetwork, error) {
		topo := poly.CNNTopologySeed(fmt.Sprintf("nf-spectrum-cnn%d", dim), []poly.CNNSpec{spec})
		topo ^= uint64(idx + 1)
		m, err := poly.BuildCNNManifest(topo, []poly.CNNSpec{spec}, []string{dtName})
		if err != nil {
			return nil, err
		}
		return poly.BuildCNNVolumetricFromManifest(m)
	}
	base := poly.CNNDemoInput(spec)
	if base == nil {
		return spectrumTask{}, fmt.Errorf("cnn%d: nil demo input", dim)
	}
	// Probe output shape once.
	net, err := factory(0)
	if err != nil {
		return spectrumTask{}, err
	}
	poly.WireNetworkLayers(net)
	net.ReleaseFP32MasterWhenIdle = false
	_ = poly.ConfigureNetworkForMode(net, poly.TrainingModeCPUMC)
	net.EnsureTrainingWeights()
	out, _, _ := poly.ForwardPolymorphic(net, base)
	if out == nil || len(out.Data) == 0 {
		return spectrumTask{}, fmt.Errorf("cnn%d: probe nil", dim)
	}
	outN := len(out.Data)
	outShape := append([]int(nil), out.Shape...)

	batches := make([]poly.TrainingBatch[float32], nBatches)
	for i := range batches {
		in := base.Clone()
		for j := range in.Data {
			in.Data[j] = float32(math.Sin(float64(i*17+j+1)*0.13))*0.5 + float32((i+j)%5)/10
		}
		tgt := make([]float32, outN)
		for j := range tgt {
			// Project input energy into output dims — easier than random sines.
			tgt[j] = in.Data[j%len(in.Data)] * 0.5
		}
		batches[i] = poly.TrainingBatch[float32]{
			Input:  in,
			Target: poly.NewTensorFromSlice(tgt, outShape...),
		}
	}
	return spectrumTask{Factory: factory, Batches: batches, LossType: "mse"}, nil
}

// repeatedProbeFitBatches: fixed I/O map (showcase-style) repeated across shards.
func repeatedProbeFitBatches(factory poly.NetworkFactory, inData []float32, n int) ([]poly.TrainingBatch[float32], error) {
	in := poly.NewTensorFromSlice(inData, 1, len(inData))
	net, err := factory(0)
	if err != nil {
		return nil, err
	}
	poly.WireNetworkLayers(net)
	net.ReleaseFP32MasterWhenIdle = false
	_ = poly.ConfigureNetworkForMode(net, poly.TrainingModeCPUMC)
	net.EnsureTrainingWeights()
	out, _, _ := poly.ForwardPolymorphic(net, in)
	if out == nil || len(out.Data) == 0 {
		return nil, fmt.Errorf("probe forward returned nil/empty")
	}
	tgtData := make([]float32, len(out.Data))
	for j := range tgtData {
		tgtData[j] = float32(math.Sin(float64(j+1) * 0.41))
	}
	tgt := poly.NewTensorFromSlice(tgtData, out.Shape...)
	batches := make([]poly.TrainingBatch[float32], n)
	for i := range batches {
		batches[i] = poly.TrainingBatch[float32]{
			Input:  in.Clone(),
			Target: tgt.Clone(),
		}
	}
	return batches, nil
}

// variedVectorFitBatches: learnable near-identity map (micro-fit, not random sin chase).
func variedVectorFitBatches(factory poly.NetworkFactory, inDim, n int) ([]poly.TrainingBatch[float32], error) {
	net, err := factory(0)
	if err != nil {
		return nil, err
	}
	poly.WireNetworkLayers(net)
	net.ReleaseFP32MasterWhenIdle = false
	_ = poly.ConfigureNetworkForMode(net, poly.TrainingModeCPUMC)
	net.EnsureTrainingWeights()
	probeIn := poly.NewTensorFromSlice(vec(inDim), 1, inDim)
	out, _, _ := poly.ForwardPolymorphic(net, probeIn)
	if out == nil || len(out.Data) == 0 {
		return nil, fmt.Errorf("probe forward returned nil/empty")
	}
	outN := len(out.Data)
	outShape := append([]int(nil), out.Shape...)

	batches := make([]poly.TrainingBatch[float32], n)
	for i := range batches {
		x := make([]float32, inDim)
		for j := range x {
			x[j] = float32((i*3+j)%11)/11
		}
		tgt := make([]float32, outN)
		for j := range tgt {
			// Scaled input + tiny bias — constant bias alone is learnable if tile is hard.
			tgt[j] = x[j%inDim]*0.5 + 0.25
		}
		batches[i] = poly.TrainingBatch[float32]{
			Input:  poly.NewTensorFromSlice(x, 1, inDim),
			Target: poly.NewTensorFromSlice(tgt, outShape...),
		}
	}
	return batches, nil
}

func repeatDtype(name string, n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = name
	}
	return out
}

func vec(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 0.01 * float32((i+1)%9)
	}
	return out
}

// microClassBatches: easy 4-way class from dominant feature block (learnable to ~90%+).
// Labels cycle; shards see mostly one class each when K % classes == 0 — fine for oracle coverage.
func microClassBatches(n, inDim, outDim int) []poly.TrainingBatch[float32] {
	if outDim < 2 {
		outDim = 2
	}
	batches := make([]poly.TrainingBatch[float32], n)
	for i := range batches {
		label := i % outDim
		x := make([]float32, inDim)
		// Near one-hot class cue in the first outDim dims + weak fillers.
		for j := 0; j < inDim; j++ {
			x[j] = 0.02 * float32((i+j)%3)
		}
		if label < inDim {
			x[label] = 1.5
		}
		y := make([]float32, outDim)
		y[label] = 1
		batches[i] = poly.TrainingBatch[float32]{
			Input:  poly.NewTensorFromSlice(x, 1, inDim),
			Target: poly.NewTensorFromSlice(y, 1, outDim),
		}
	}
	return batches
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
