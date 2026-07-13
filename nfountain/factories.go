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

func spectrumFactoryAndBatches(family string, dt poly.DType, k int) (poly.NetworkFactory, []poly.TrainingBatch[float32], error) {
	dtName := dt.String()
	nBatches := max(8, k*4)

	switch family {
	case "dense":
		sizes := []int{8, 12, 6, 4}
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
		return factory, syntheticVectorBatches(nBatches, sizes[0], sizes[len(sizes)-1]), nil

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
		batches, err := probeIOBatches(factory, vec(8), nBatches)
		return factory, batches, err

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
		batches, err := probeIOBatches(factory, vec(8), nBatches)
		return factory, batches, err

	case "cnn1":
		return cnnFamily(1, 8, dtName, nBatches)
	case "cnn2":
		return cnnFamily(2, 8, dtName, nBatches)
	case "cnn3":
		return cnnFamily(3, 4, dtName, nBatches)

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
		batches, err := probeIOBatches(factory, vec(4), nBatches)
		return factory, batches, err

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
		batches, err := probeIOBatches(factory, vec(4), nBatches)
		return factory, batches, err

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
				tok.Data[j] = float32((i + j) % spec.VocabSize)
			}
			batches[i] = poly.TrainingBatch[float32]{
				Input:  tok,
				Target: poly.NewTensorFromSlice(sinVec(outDim), 1, outDim),
			}
		}
		return factory, batches, nil

	case "residual":
		spec := poly.ResidualSpec{In: 4, Out: 4}
		factory := func(idx int) (*poly.VolumetricNetwork, error) {
			topo := poly.ResidualTopologySeed("nf-spectrum-res", spec)
			topo ^= uint64(idx + 1)
			m, err := poly.BuildResidualManifest(topo, spec, dtName)
			if err != nil {
				return nil, err
			}
			return poly.BuildResidualVolumetricFromManifest(m)
		}
		batches, err := probeIOBatches(factory, vec(4), nBatches)
		return factory, batches, err
	}

	return nil, nil, fmt.Errorf("unknown family %q", family)
}

func cnnFamily(dim, spatial int, dtName string, nBatches int) (poly.NetworkFactory, []poly.TrainingBatch[float32], error) {
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
	in := poly.CNNDemoInput(spec)
	if in == nil {
		return nil, nil, fmt.Errorf("cnn%d: nil demo input", dim)
	}
	batches, err := probeIOBatchesWithInput(factory, in, nBatches)
	return factory, batches, err
}

// probeIOBatches builds one net, runs ForwardPolymorphic, and makes MSE targets of matching shape.
func probeIOBatches(factory poly.NetworkFactory, inData []float32, n int) ([]poly.TrainingBatch[float32], error) {
	in := poly.NewTensorFromSlice(inData, 1, len(inData))
	return probeIOBatchesWithInput(factory, in, n)
}

func probeIOBatchesWithInput(factory poly.NetworkFactory, in *poly.Tensor[float32], n int) ([]poly.TrainingBatch[float32], error) {
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
	tgt := poly.NewTensorFromSlice(sinVec(len(out.Data)), out.Shape...)
	batches := make([]poly.TrainingBatch[float32], n)
	for i := range batches {
		batches[i] = poly.TrainingBatch[float32]{
			Input:  in.Clone(),
			Target: tgt.Clone(),
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

func sinVec(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(float64(i+1) * 0.41))
	}
	return out
}

func syntheticVectorBatches(n, inDim, outDim int) []poly.TrainingBatch[float32] {
	batches := make([]poly.TrainingBatch[float32], n)
	for i := range batches {
		x := make([]float32, inDim)
		y := make([]float32, outDim)
		for j := range x {
			x[j] = float32((i+j)%7) / 7
		}
		y[i%outDim] = 1
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
