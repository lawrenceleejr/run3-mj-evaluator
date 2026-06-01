# run3-mj-evaluator

ML evaluator for the 6-jet multijet analysis. Reads a slimmed ROOT file
(produced by `run3-mj-slimmer/slim.py`), runs one or more ONNX neural-net
models to predict which 3+3 jets belong to each pair-produced particle, and
writes a new ROOT file with the original branches intact plus new
`MLCandidate` collections.

## Pipeline position

```
ScoutingNanoAOD ROOT  →  slim.py  →  slimmed ROOT  →  evaluate.py  →  evaluated ROOT
```

## Usage

```bash
python evaluate.py input.root output.root config.json
python evaluate.py input.root output.root config.json --tree events --chunk-size 50000
```

## Config format

```json
{
    "metadata": {"version": "v1"},
    "models": [
        {
            "type":         "spanet",
            "path":         "models/cart_model.onnx",
            "label":        "SPANet",
            "input_format": "cart"
        },
        {
            "type":       "comb_solver",
            "path":       "models/best_model.onnx",
            "label":      "CombSolver",
            "normalized": true
        }
    ]
}
```

### Model types

| `type` | Description | Required extra keys |
|---|---|---|
| `spanet` | SPANet ONNX model — batched inference, takes `(N, J, 4)` source + `(N, J)` mask | `input_format`: `"cart"` (px,py,pz,e) or `"spher"` (pt,eta,phi,e) |
| `comb_solver` | CombinatorialSolver ONNX — run one event at a time (batch=1 baked into graph), takes top-7 pT jets as `(1, 7, 4)` (E,px,py,pz) | `normalized`: `true` (divide by HT) or `false` (raw units) |

## Output ROOT file structure

### `events` TTree

All original branches from the slimmed file are copied through, plus for
each model in the config:

| Branch | Type | Description |
|---|---|---|
| `{label}Candidate_pt[2]` | float32 | pT of the two reconstructed trijet particles |
| `{label}Candidate_eta[2]` | float32 | η |
| `{label}Candidate_phi[2]` | float32 | φ |
| `{label}Candidate_mass[2]` | float32 | Invariant mass [GeV] |
| `{label}Candidate_jetIdx0[2]` | int32 | Index of 1st assigned jet in `ScoutingPFJet_*` arrays |
| `{label}Candidate_jetIdx1[2]` | int32 | Index of 2nd assigned jet |
| `{label}Candidate_jetIdx2[2]` | int32 | Index of 3rd assigned jet |

Each array has two elements per event: `[0]` = candidate 1, `[1]` = candidate 2.
The jet indices point into the per-event `ScoutingPFJet_*` arrays so you can
recover the constituent jet kinematics.

### `meta` TTree

One-row tree with evaluator version, config path, input file path, and
semicolon-joined model paths/labels.

### `cutflow` histogram

Passed through from the input file unchanged (when present).

## Dependencies

```
uproot >= 5
awkward >= 2
onnxruntime
numpy
```

The slimmed input file must contain at least `ScoutingPFJet_pt`,
`ScoutingPFJet_eta`, `ScoutingPFJet_phi`, and either
`ScoutingPFJet_px/py/pz/e` (preferred) or `ScoutingPFJet_m`.
`slim.py` computes and stores `px/py/pz/e` automatically.
