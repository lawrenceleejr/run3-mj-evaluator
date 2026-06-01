#!/usr/bin/env python3
"""evaluate.py - ML evaluator for the 6-jet multijet analysis.

Reads a slimmed ROOT file (produced by slim.py), runs one or more ONNX models
on per-event jet 4-vectors, and writes a new ROOT file containing:
  - All original branches (pass-through)
  - For each model: {label}Candidate_pt[2], eta[2], phi[2], mass[2],
    jetIdx0[2], jetIdx1[2], jetIdx2[2]  — the two predicted trijet particles
  - TH1 'version': config metadata.version string
  - TH1 'cutflow': passed through from the input file if present

Usage:
    python evaluate.py input.root output.root config.json
    python evaluate.py input.root output.root config.json --tree events --chunk-size 50000

Config JSON format:
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
"""

import argparse
import json
import sys
import warnings
from itertools import combinations

import awkward as ak
import boost_histogram as bh
import numpy as np
import onnxruntime
import uproot

_JET_BRANCH = "ScoutingPFJet"

_VALID_TYPES   = {"spanet", "comb_solver"}
_VALID_FORMATS = {"cart", "spher"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path):
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Config file not found: {config_path}")
    except json.JSONDecodeError as exc:
        sys.exit(f"Invalid JSON in {config_path}: {exc}")

    if "models" not in cfg or not cfg["models"]:
        sys.exit("Config must contain a non-empty 'models' list.")

    for i, m in enumerate(cfg["models"]):
        for key in ("type", "path", "label"):
            if key not in m:
                sys.exit(f"Model {i} is missing required key '{key}'.")
        if m["type"] not in _VALID_TYPES:
            sys.exit(f"Model {i} unknown type '{m['type']}'. Valid: {_VALID_TYPES}")
        if m["type"] == "spanet":
            if m.get("input_format") not in _VALID_FORMATS:
                sys.exit(f"Model {i} (spanet) 'input_format' must be 'cart' or 'spher'.")
        if m["type"] == "comb_solver" and "normalized" not in m:
            sys.exit(f"Model {i} (comb_solver) must have 'normalized': true or false.")

    return cfg


def _label_to_prefix(label):
    """Sanitize a model label into a valid ROOT branch prefix."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in label)


# ---------------------------------------------------------------------------
# Jagged → padded numpy helpers
# ---------------------------------------------------------------------------

def _padded(arr, max_len, fill=0.0):
    """Pad a jagged awkward array to shape (N, max_len) with a scalar fill."""
    return ak.to_numpy(ak.fill_none(ak.pad_none(arr, max_len, axis=1, clip=True), fill))


def _mask_from_lengths(lengths, max_len):
    """Boolean (N, max_len) mask: True for valid jet slots."""
    return np.arange(max_len)[None, :] < np.asarray(lengths)[:, None]


def chunk_to_numpy(chunk):
    """Convert one uproot chunk to padded (N, J) float64 arrays + bool mask."""
    jets   = chunk[_JET_BRANCH]
    n_jets = ak.to_numpy(ak.num(jets["pt"], axis=1))
    max_j  = int(n_jets.max()) if len(n_jets) > 0 else 0

    mask = _mask_from_lengths(n_jets, max_j)
    pt   = _padded(jets["pt"],  max_j)
    eta  = _padded(jets["eta"], max_j)
    phi  = _padded(jets["phi"], max_j)
    m    = _padded(jets["m"],   max_j)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e  = np.sqrt(px**2 + py**2 + pz**2 + m**2)

    return pt, eta, phi, px, py, pz, e, mask


# ---------------------------------------------------------------------------
# SPANet
# ---------------------------------------------------------------------------

def _extract_triplets(assignments):
    """Argmax over (J, J, J) assignment tensor. assignments: (B, J, J, J) → (B, 3)."""
    B = assignments.shape[0]
    out = np.zeros((B, 3), dtype=int)
    for b in range(B):
        out[b] = np.unravel_index(np.argmax(assignments[b]), assignments[b].shape)
    return out


def run_spanet(session, source, mask):
    """Run a SPANet session on a batch.

    source : (N, J, 4) float32
    mask   : (N, J) bool  — True for valid jets
    Returns t1_idx, t2_idx each (N, 3) int — indices into the J-jet array.
    """
    t1_assign, t2_assign, _, _ = session.run(
        None,
        {"source_data": source, "source_mask": mask},
    )
    return _extract_triplets(t1_assign), _extract_triplets(t2_assign)


# ---------------------------------------------------------------------------
# CombinatorialSolver
# ---------------------------------------------------------------------------

def _build_7jet_assignment_tables():
    """(70, 3) index arrays for the 7-jet CombinatorialSolver output ordering.

    Output logits have shape (batch, 70) = 7 ISR choices x 10 unique 3+3 partitions
    of the remaining 6 jets. Ordering matches CombinatorialSolver's
    ``enumerate_assignments(7)``: outer loop over ISR jet, inner over groupings.
    """
    g1_list, g2_list = [], []
    for isr in range(7):
        remaining = [j for j in range(7) if j != isr]
        seen = set()
        for g1 in combinations(remaining, 3):
            g2    = tuple(j for j in remaining if j not in g1)
            canon = (min(g1, g2), max(g1, g2))
            if canon not in seen:
                seen.add(canon)
                g1_list.append(list(canon[0]))
                g2_list.append(list(canon[1]))
    return np.array(g1_list, dtype=int), np.array(g2_list, dtype=int)


_COMB_G1, _COMB_G2 = _build_7jet_assignment_tables()  # each (70, 3)


def prepare_comb_input(pt, eta, phi, e, px, py, pz, mask):
    """Select top-7 pT jets and build (N, 7, 4) inputs for CombinatorialSolver.

    Returns
    -------
    input_norm : (N, 7, 4) float32  — HT-normalised (E, px, py, pz) for the ML model
    input_raw  : (N, 7, 4) float32  — physical units for the classical solver
    spher_7jet : (N, 7, 4) float64  — (pt, eta, phi, e) in original units
    top7_idx   : (N, 7) int         — positions in the original J-jet array
    """
    N = pt.shape[0]
    ht = np.where(mask, pt, 0.0).sum(axis=1, keepdims=True).clip(min=1e-6)

    pt_masked = np.where(mask, pt, -np.inf)
    top7_idx  = np.argsort(-pt_masked, axis=1)[:, :7]  # (N, 7)

    rows = np.arange(N)[:, None]
    e7   = e  [rows, top7_idx]
    px7  = px [rows, top7_idx]
    py7  = py [rows, top7_idx]
    pz7  = pz [rows, top7_idx]
    pt7  = pt [rows, top7_idx]
    eta7 = eta[rows, top7_idx]
    phi7 = phi[rows, top7_idx]

    epxpypz    = np.stack([e7, px7, py7, pz7], axis=-1)
    input_norm = (epxpypz / ht[:, :, None]).astype(np.float32)
    input_raw  = epxpypz.astype(np.float32)
    spher_7jet = np.stack([pt7, eta7, phi7, e7], axis=-1)

    return input_norm, input_raw, spher_7jet, top7_idx


def run_comb_solver(session, model_input):
    """Run a CombinatorialSolver session one event at a time (batch_size=1 baked in ONNX).

    model_input : (N, 7, 4) float32
    Returns t1_idx, t2_idx each (N, 3) int — indices into the 7-jet array.
    """
    N    = model_input.shape[0]
    best = np.empty(N, dtype=int)
    for i in range(N):
        (logits,) = session.run(None, {"four_momenta": model_input[i : i + 1]})
        best[i]   = np.argmax(logits)
    return _COMB_G1[best], _COMB_G2[best]


# ---------------------------------------------------------------------------
# Candidate 4-vector reconstruction
# ---------------------------------------------------------------------------

def candidate_fourvec(pt, eta, phi, e, indices):
    """Sum 3 jets specified by indices into a candidate 4-vector.

    pt, eta, phi, e : (N, J) float64
    indices         : (N, 3) int

    Returns pt_c, eta_c, phi_c, mass_c each (N,) float64.
    """
    rows = np.arange(len(indices))[:, None]
    px3  = (pt[rows, indices] * np.cos(phi[rows, indices])).sum(1)
    py3  = (pt[rows, indices] * np.sin(phi[rows, indices])).sum(1)
    pz3  = (pt[rows, indices] * np.sinh(eta[rows, indices])).sum(1)
    e3   = e[rows, indices].sum(1)

    m2   = e3**2 - (px3**2 + py3**2 + pz3**2)
    mass = np.sqrt(np.maximum(m2, 0.0))
    pt_c = np.sqrt(px3**2 + py3**2)
    phi_c = np.arctan2(py3, px3)
    eta_c = np.arcsinh(pz3 / np.where(pt_c > 0, pt_c, 1e-10))

    return pt_c, eta_c, phi_c, mass


# ---------------------------------------------------------------------------
# ONNX session loader
# ---------------------------------------------------------------------------

def _load_session(model_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return onnxruntime.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(input_path, output_path, config, config_path, in_tree_name, chunk_size):
    models  = config["models"]
    version = config.get("metadata", {}).get("version", "unknown")

    print(f"Input:   {input_path}  (tree: {in_tree_name})")
    print(f"Output:  {output_path}")
    print(f"Version: {version}")
    print(f"Models:  {[m['label'] for m in models]}")
    print()

    print("Loading ONNX sessions...")
    sessions = []
    for m in models:
        print(f"  {m['label']} ({m['type']})  <-  {m['path']}")
        sessions.append(_load_session(m["path"]))
    print()

    with uproot.open(input_path) as in_file:
        if in_tree_name not in in_file:
            sys.exit(
                f"Tree '{in_tree_name}' not found in {input_path}. "
                f"Available keys: {list(in_file.keys())}"
            )

        tree      = in_file[in_tree_name]
        tree_keys = set(tree.keys())

        # The slimmer writes a nested ScoutingPFJet struct; its sub-branches
        # appear in tree.keys() as "ScoutingPFJet.pt", "ScoutingPFJet.eta", …
        if f"{_JET_BRANCH}.pt" not in tree_keys:
            sys.exit(
                f"Expected nested branch '{_JET_BRANCH}.pt' not found in tree "
                f"'{in_tree_name}'. Available keys: {sorted(tree_keys)[:20]}"
            )

        total_entries = tree.num_entries
        n_chunks      = max(1, (total_entries + chunk_size - 1) // chunk_size)
        print(f"Tree has {total_entries:,} events -> {n_chunks} chunk(s) of up to {chunk_size:,}")
        print()

        total_in = total_out = 0

        with uproot.recreate(output_path) as out_file:
            out_tree  = None
            chunk_num = 0

            for chunk in tree.iterate(library="ak", step_size=chunk_size):
                chunk_num += 1
                n_chunk    = len(chunk)
                total_in  += n_chunk
                print(
                    f"[{chunk_num}/{n_chunks}] Chunk of {n_chunk:,} events"
                    f"  (running total: {total_in:,}/{total_entries:,})"
                )

                print("  Converting branches to numpy arrays...", flush=True)
                pt, eta, phi, px, py, pz, e, mask = chunk_to_numpy(chunk)

                # CombSolver uses the top-7 pT jets; SPANet uses the top-6 (first 6
                # of the same sorted list). Compute once and share across all models.
                comb_norm, comb_raw, _spher_7j, top7_idx = prepare_comb_input(
                    pt, eta, phi, e, px, py, pz, mask
                )
                rows7    = np.arange(n_chunk)[:, None]
                top6_idx = top7_idx[:, :6]  # top-6 is a subset of top-7
                pt6  = pt [rows7, top6_idx]
                eta6 = eta[rows7, top6_idx]
                phi6 = phi[rows7, top6_idx]
                e6   = e  [rows7, top6_idx]
                px6  = px [rows7, top6_idx]
                py6  = py [rows7, top6_idx]
                pz6  = pz [rows7, top6_idx]
                # Slimmer guarantees ≥6 jets, so the top-6 mask is all-True.
                mask6 = np.ones((n_chunk, 6), dtype=bool)

                out_record = {}

                # Pass through all original top-level branches (preserves nested
                # ScoutingPFJet struct from the slimmer).
                print("  Copying input branches...", flush=True)
                for branch in ak.fields(chunk):
                    out_record[branch] = chunk[branch]

                for model_cfg, session in zip(models, sessions):
                    prefix = _label_to_prefix(model_cfg["label"])
                    mtype  = model_cfg["type"]

                    print(f"  Running {model_cfg['label']} ({mtype})...", flush=True)

                    if mtype == "spanet":
                        if model_cfg["input_format"] == "cart":
                            source = np.stack([px6, py6, pz6, e6], axis=-1).astype(np.float32)
                        else:
                            source = np.stack([pt6, eta6, phi6, e6], axis=-1).astype(np.float32)
                        t1_6, t2_6 = run_spanet(session, source, mask6)
                        t1_idx = top6_idx[rows7, t1_6]  # (N, 3) into full jet array
                        t2_idx = top6_idx[rows7, t2_6]

                    elif mtype == "comb_solver":
                        comb_in = comb_norm if model_cfg["normalized"] else comb_raw
                        t1_7, t2_7 = run_comb_solver(session, comb_in)
                        t1_idx = top7_idx[rows7, t1_7]  # (N, 3) into full jet array
                        t2_idx = top7_idx[rows7, t2_7]

                    pt1, eta1, phi1, m1 = candidate_fourvec(pt, eta, phi, e, t1_idx)
                    pt2, eta2, phi2, m2 = candidate_fourvec(pt, eta, phi, e, t2_idx)

                    # Store as (N, 2) fixed-size arrays: index 0 = candidate 1, index 1 = candidate 2
                    out_record[f"{prefix}Candidate_pt"]      = ak.Array(np.stack([pt1,          pt2         ], axis=1).astype(np.float32))
                    out_record[f"{prefix}Candidate_eta"]     = ak.Array(np.stack([eta1,         eta2        ], axis=1).astype(np.float32))
                    out_record[f"{prefix}Candidate_phi"]     = ak.Array(np.stack([phi1,         phi2        ], axis=1).astype(np.float32))
                    out_record[f"{prefix}Candidate_mass"]    = ak.Array(np.stack([m1,           m2          ], axis=1).astype(np.float32))
                    out_record[f"{prefix}Candidate_jetIdx0"] = ak.Array(np.stack([t1_idx[:, 0], t2_idx[:, 0]], axis=1).astype(np.int32))
                    out_record[f"{prefix}Candidate_jetIdx1"] = ak.Array(np.stack([t1_idx[:, 1], t2_idx[:, 1]], axis=1).astype(np.int32))
                    out_record[f"{prefix}Candidate_jetIdx2"] = ak.Array(np.stack([t1_idx[:, 2], t2_idx[:, 2]], axis=1).astype(np.int32))

                total_out += n_chunk

                print(f"  Writing {n_chunk:,} events to output...", flush=True)
                if out_tree is None:
                    out_file.mktree(
                        "events",
                        {name: arr.type for name, arr in out_record.items()},
                    )
                    out_tree = out_file["events"]
                out_tree.extend(out_record)
                print(f"  Chunk {chunk_num}/{n_chunks} complete.")
                print()

            # Version histogram (StrCategory is the only reliable way to store
            # strings in uproot; byte-string TTree branches trigger RNTuple routing).
            version_hist = bh.Histogram(bh.axis.StrCategory([version]), storage=bh.storage.Double())
            version_hist.view()[0] = 1.0
            out_file["version"] = version_hist

            # Pass through cutflow histogram if present
            if "cutflow" in in_file:
                print("Copying cutflow histogram from input...")
                out_file["cutflow"] = in_file["cutflow"]

    print(f"Done.  {total_in:,} events processed -> {total_out:,} written.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run ONNX model(s) on slimmed ROOT jets and write MLCandidate "
            "objects (reconstructed trijet particles) to a new ROOT file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",  help="Input slimmed ROOT file (from slim.py)")
    parser.add_argument("output", help="Output ROOT file")
    parser.add_argument("config", help="JSON config file specifying models to run")
    parser.add_argument(
        "--tree", default="events", metavar="NAME",
        help="Input tree name",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50_000, metavar="N",
        help="Events per processing chunk",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(
        input_path=args.input,
        output_path=args.output,
        config=cfg,
        config_path=args.config,
        in_tree_name=args.tree,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
