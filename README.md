# loom_neural_fountain

**Neural Fountain** demo: train K specialist nets on MNIST shards, ship their weights through an LT fountain, peel them back **byte-exact**, then run a **Master** ensemble — **no layer_seed search**.

```bash
cd loom_neural_fountain
./run.sh                 # full 56k train / 14k val · K=16
./run.sh quick           # 4k subset · K=8 smoke
go run . -k 32 -epochs 8
```

Uses `../loom_seed_mnist/data` if present; otherwise downloads MNIST into `./data`.

Core library: [`poly.NeuralFountain`](../loom/poly/neural_fountain.go) (any model / data).  
MNIST helpers: [`github.com/openfluke/loom/neural`](../loom/neural) façade.

---

## What this is

Pixel fountain (`loom_fountain_codes`) recovers **K MNIST image blocks** to 100% via spray / peel.

Neural Fountain does the **same LT algebra** on **K specialist weight blobs**:

| Step | What happens |
|------|----------------|
| **1. Data** | All 70k MNIST → shuffle → **80/20** (56k train / 14k val) |
| **2. Shards** | Split train into **K** piles so every train sample is covered |
| **3. Specialize** | `poly.Train` a dense net `784→128→64→10` on each shard |
| **4. Pack** | Flatten each specialist’s FP32 weights → one fountain source block (~427 KiB) |
| **5. Fountain** | LT XOR spray + ~30% loss → peel until **K/K recovered byte-exact** |
| **6. Master** | Recovered experts + **average logits** at inference |

Fountain here is **transport / reassembly**, not the teacher. Learning happens in the specialist `poly.Train` steps.

---

## How to read the scores

From a full `./run.sh` (K=16, 5 epochs/shard):

```text
oracle train coverage   ≈ 97.8%   ← each train point scored by its shard’s expert
ensemble train          ≈ 95.4%   ← Master averages all specialists
ensemble val            ≈ 94.8%   ← real holdout
recovered specialists   = 16/16   ← fountain byte-exact (same event as “bucket full”)
```

- **Oracle coverage** ≈ “did we cover the whole train set’s knowledge?” — each sample hits the expert that saw it. This is the neural analogue of recovering all train blocks.
- **Ensemble val** ≈ deployable Master quality (no oracle; average logits).
- **DeviationMetrics** on val = real digit prediction heatmaps (predicted class vs label).

Specialists alone usually hit ~97–98% **on their own shard**; the Master is how you use them together after the fountain.

---

## What this is not

- Not seed training / DNA search (`loom_seed_mnist`).
- Not “fountain XOR invents a net without backprop.”
- Not one global SGD over all 56k in a single model (though you could distill the Master later).
- Weight-mean of specialists is *not* used as the primary Master (averaging weights of differently trained nets tends toward chance).

---

## Knobs

| Flag / env | Default | Meaning |
|------------|---------|---------|
| *(none)* | K=16, 5 epochs | full 80/20 |
| `./run.sh quick` / `LOOM_NEURAL_FOUNTAIN_QUICK=1` | K=8, 4k train | fast smoke |
| `-k N` | 16 | number of specialists / shards |
| `-epochs N` | 5 | `poly.Train` epochs per specialist |
| `-loss` | 0.3 | fountain drop erase rate |
| `[dataDir]` | auto | MNIST gzip directory |

More specialists → smaller shards → easier shard memorization → higher oracle coverage, larger fountain payload. Fewer specialists → each net sees more data, slower specialize phase.

---

## Generic use (any net / any data)

```go
// Your architecture — dense, CNN, residual, …
factory := func(i int) (*poly.VolumetricNetwork, error) {
    // build a fresh specialist with identical weight layout
    return buildMyNet(i)
}
// or: factory := poly.DenseSpecialistFactory("name", sizes, nil)

master, err := poly.NeuralFountain(factory, batches, poly.DefaultNeuralFountainConfig())
out, err := master.Forward(input) // average specialist outputs
```

This MNIST demo is just one factory + `TrainingBatch` source.

---

## Relation to pixel fountain

```text
loom_fountain_codes   → recover MNIST pixels
loom_neural_fountain  → recover specialist weights → Master classifier
```

Same spray / peel completion event (`K/K` byte-exact). Different cargo: images vs nets.
