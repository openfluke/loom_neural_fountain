package nfountain

import (
	"math"

	"github.com/openfluke/loom/poly"
)

// evalOracleEnsembleAcc scores classification after fountain recover/consolidate.
func evalOracleEnsembleAcc(master *poly.FountainMaster, batches []poly.TrainingBatch[float32]) (oracle, ens float64) {
	if master == nil || len(batches) == 0 {
		return 0, 0
	}
	var oOK, eOK, n int
	for i, b := range batches {
		if b.Input == nil || b.Target == nil || len(b.Target.Data) == 0 {
			continue
		}
		want := argmax(b.Target.Data)
		if pred, err := master.OracleArgmax(i, b.Input); err == nil && pred == want {
			oOK++
		}
		if pred, err := master.ForwardArgmax(b.Input); err == nil && pred == want {
			eOK++
		}
		n++
	}
	if n == 0 {
		return 0, 0
	}
	return float64(oOK) / float64(n), float64(eOK) / float64(n)
}

// evalOracleEnsembleFit: 100-scale fit score from MSE vs zero-baseline (regression families).
// 100 ≈ ensemble matches targets; 0 ≈ no better than predicting zeros.
func evalOracleEnsembleFit(master *poly.FountainMaster, batches []poly.TrainingBatch[float32]) (oracleFit, ensFit, ensMSE float64) {
	if master == nil || len(batches) == 0 {
		return 0, 0, 0
	}
	var oMSE, eMSE, base float64
	var n int
	for i, b := range batches {
		if b.Input == nil || b.Target == nil || len(b.Target.Data) == 0 {
			continue
		}
		base += mseTo(nil, b.Target.Data) // vs zeros
		if out, err := master.OracleForward(i, b.Input); err == nil && out != nil {
			oMSE += mseTo(out.Data, b.Target.Data)
		} else {
			oMSE += mseTo(nil, b.Target.Data)
		}
		if out, err := master.Forward(b.Input); err == nil && out != nil {
			eMSE += mseTo(out.Data, b.Target.Data)
		} else {
			eMSE += mseTo(nil, b.Target.Data)
		}
		n++
	}
	if n == 0 {
		return 0, 0, 0
	}
	oMSE /= float64(n)
	eMSE /= float64(n)
	base /= float64(n)
	ensMSE = eMSE
	return fitScore(oMSE, base), fitScore(eMSE, base), ensMSE
}

func fitScore(mse, baseline float64) float64 {
	if math.IsNaN(mse) || math.IsInf(mse, 0) {
		return 0
	}
	if baseline <= 1e-12 || math.IsNaN(baseline) || math.IsInf(baseline, 0) {
		if mse <= 1e-12 {
			return 100
		}
		return 0
	}
	s := (1 - mse/baseline) * 100
	if s < 0 {
		return 0
	}
	if s > 100 {
		return 100
	}
	return s
}

func mseTo(pred, tgt []float32) float64 {
	if len(tgt) == 0 {
		return 0
	}
	var s float64
	if len(pred) == 0 {
		for _, t := range tgt {
			s += float64(t) * float64(t)
		}
		return s / float64(len(tgt))
	}
	n := len(tgt)
	if len(pred) < n {
		n = len(pred)
	}
	for i := 0; i < n; i++ {
		d := float64(pred[i] - tgt[i])
		s += d * d
	}
	return s / float64(len(tgt))
}

func argmax(v []float32) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

func qualityOK(classify bool, oraclePct, ensPct, minPct float64) bool {
	if minPct <= 0 {
		return true
	}
	if math.IsNaN(oraclePct) {
		oraclePct = 0
	}
	if math.IsNaN(ensPct) {
		ensPct = 0
	}
	if classify {
		return oraclePct >= minPct
	}
	// Regression / fit: Master ensemble is the consolidate deployable metric.
	return math.Max(oraclePct, ensPct) >= minPct
}

func dtypeQualityFloor(dt poly.DType, base float64) float64 {
	// Low-bit / extreme quantization: softer floor (still learned, not random).
	switch dt {
	case poly.DTypeBinary, poly.DTypeTernary, poly.DTypeInt2, poly.DTypeUint2,
		poly.DTypeInt4, poly.DTypeUint4, poly.DTypeFP4,
		poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		return math.Min(base, 70)
	default:
		return base
	}
}
