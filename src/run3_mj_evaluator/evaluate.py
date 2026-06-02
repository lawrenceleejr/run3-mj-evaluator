#!/usr/bin/env python3
"""evaluate.py - ML evaluator for the 6-jet multijet analysis.

Reads a slimmed ROOT file (produced by slim.py), runs one or more ONNX models
on per-event jet 4-vectors, and writes a new ROOT file containing:
  - All original branches (pass-through), including a GenJet collection if the
    input is MC and contains one.
  - For each model: {label}Candidate_pt[2], eta[2], phi[2], mass[2],
    jetIdx0[2], jetIdx1[2], jetIdx2[2]  — the two predicted trijet particles.
  - With --run-on-genjets: additional {label}GenCandidate_* branches produced
    by running each model on the GenJet collection (indices point into GenJet).
  - TH1 'version': config metadata.version string
  - TH1 'cutflow': passed through from the input file if present

Usage:
    python evaluate.py input.root output.root config.json
    python evaluate.py input.root output.root config.json --tree events --chunk-size 50000
    python evaluate.py mc.root out.root config.json --run-on-genjets

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
import os
import sys
import time
import warnings
from itertools import combinations

import awkward as ak
import boost_histogram as bh
import numpy as np
import onnxruntime
import uproot

_JET_BRANCH      = "ScoutingPFJet"
_GEN_JET_BRANCH  = "GenJet"
_GEN_PART_BRANCH = "GenPart"

_VALID_TYPES   = {"spanet", "comb_solver"}
_VALID_FORMATS = {"cart", "spher"}


# ---------------------------------------------------------------------------
# Timing / progress helpers
# ---------------------------------------------------------------------------

def _fmt_duration(seconds):
    """Format a duration in seconds as a compact human-readable string."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f}us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f}ms"
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    m, s = divmod(seconds, 60.0)
    if m < 60.0:
        return f"{int(m)}m{s:04.1f}s"
    h, m = divmod(int(m), 60)
    return f"{h}h{int(m):02d}m{s:04.1f}s"


class _Timer:
    """Context manager: prints '<msg>... <duration>' on exit."""
    def __init__(self, msg, indent="  "):
        self.msg    = msg
        self.indent = indent
        self.t0     = None
        self.dt     = None

    def __enter__(self):
        print(f"{self.indent}{self.msg}...", flush=True)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.dt = time.perf_counter() - self.t0
        status = "FAILED" if exc_type is not None else "done"
        print(f"{self.indent}  -> {status} in {_fmt_duration(self.dt)}", flush=True)
        return False


def _progress_iter(total, desc, indent="    ", min_interval=0.2, width=30):
    """Yield 0..total-1, drawing an ASCII progress bar to stderr.

    Designed for tight ONNX-inference loops: cheap (only redraws every
    ``min_interval`` seconds), TTY-aware (falls back to periodic newline
    output when stderr isn't a terminal so log files stay readable).
    """
    is_tty = sys.stderr.isatty() if hasattr(sys.stderr, "isatty") else False
    t_start = time.perf_counter()
    t_last  = 0.0
    prefix  = f"{indent}{desc}"

    def _draw(i, final=False):
        elapsed = time.perf_counter() - t_start
        frac    = i / total if total > 0 else 1.0
        eta     = (elapsed / i) * (total - i) if i > 0 else 0.0
        rate    = i / elapsed if elapsed > 0 else 0.0
        if is_tty:
            filled = int(width * frac)
            bar = "#" * filled + "-" * (width - filled)
            line = (
                f"\r{prefix} [{bar}] {i:>{len(str(total))}}/{total} "
                f"({frac * 100:5.1f}%) {rate:>6.1f}/s "
                f"elapsed {_fmt_duration(elapsed)} eta {_fmt_duration(eta)}"
            )
            sys.stderr.write(line)
            if final:
                sys.stderr.write("\n")
            sys.stderr.flush()
        else:
            sys.stderr.write(
                f"{prefix} {i}/{total} ({frac * 100:.1f}%) "
                f"{rate:.1f}/s elapsed {_fmt_duration(elapsed)} "
                f"eta {_fmt_duration(eta)}\n"
            )
            sys.stderr.flush()

    for i in range(total):
        now = time.perf_counter()
        if i == 0 or (now - t_last) >= min_interval:
            _draw(i)
            t_last = now
        yield i
    _draw(total, final=True)


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
# Jet sub-branch access (nested vs flat layout)
# ---------------------------------------------------------------------------

def jet_format(keys, branch=_JET_BRANCH):
    """Determine how the jet kinematics are stored in a set of *tree* keys.

    uproot reports a tree's branches with dotted names, so the two supported
    layouts are distinguishable directly:
      - "nested": a single ``<branch>`` record whose sub-branches show up as
        ``<branch>.pt``, ``<branch>.eta``, … (dotted).
      - "flat":   NanoAOD-style separate branches ``<branch>_pt``,
        ``<branch>_eta``, … (underscored).

    Returns "nested", "flat", or None if neither is present.
    """
    keys = set(keys)
    if f"{branch}.pt" in keys:
        return "nested"
    if f"{branch}_pt" in keys:
        return "flat"
    return None


def jet_subarrays(chunk, branch=_JET_BRANCH):
    """Return (pt, eta, phi, m) jagged awkward arrays for the given collection.

    Handles both the nested ``<branch>`` record and the flat ``<branch>_<field>``
    layout transparently.
    """
    fields = set(ak.fields(chunk))
    if branch in fields and ak.fields(chunk[branch]):
        jets = chunk[branch]
        return jets["pt"], jets["eta"], jets["phi"], jets["m"]
    if f"{branch}_pt" in fields:
        return (
            chunk[f"{branch}_pt"],
            chunk[f"{branch}_eta"],
            chunk[f"{branch}_phi"],
            chunk[f"{branch}_m"],
        )
    raise KeyError(
        f"Could not find jet branches '{branch}.pt' (nested) or "
        f"'{branch}_pt' (flat) in chunk fields: {sorted(fields)}"
    )


# ---------------------------------------------------------------------------
# Jagged → padded numpy helpers
# ---------------------------------------------------------------------------

def _padded(arr, max_len, fill=0.0):
    """Pad a jagged awkward array to shape (N, max_len) with a scalar fill."""
    return ak.to_numpy(ak.fill_none(ak.pad_none(arr, max_len, axis=1, clip=True), fill))


def _mask_from_lengths(lengths, max_len):
    """Boolean (N, max_len) mask: True for valid jet slots."""
    return np.arange(max_len)[None, :] < np.asarray(lengths)[:, None]


def chunk_to_numpy(chunk, branch=_JET_BRANCH, min_jets=0):
    """Convert one uproot chunk to padded (N, J) float64 arrays + bool mask.

    ``min_jets`` forces the padded width to be at least that many slots, so
    downstream code that always expects e.g. 7 jets (CombSolver) works even on
    sparse collections like GenJet where some events have fewer entries.
    """
    jet_pt, jet_eta, jet_phi, jet_m = jet_subarrays(chunk, branch)
    n_jets = ak.to_numpy(ak.num(jet_pt, axis=1))
    max_j  = max(min_jets, int(n_jets.max()) if len(n_jets) > 0 else 0)

    mask = _mask_from_lengths(n_jets, max_j)
    pt   = _padded(jet_pt,  max_j)
    eta  = _padded(jet_eta, max_j)
    phi  = _padded(jet_phi, max_j)
    m    = _padded(jet_m,   max_j)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e  = np.sqrt(px**2 + py**2 + pz**2 + m**2)

    return pt, eta, phi, px, py, pz, e, mask


# ---------------------------------------------------------------------------
# SPANet
# ---------------------------------------------------------------------------

def _extract_triplets(assignments, progress_desc=None):
    """Argmax over (J, J, J) assignment tensor. assignments: (B, J, J, J) → (B, 3)."""
    B = assignments.shape[0]
    out = np.zeros((B, 3), dtype=int)
    iterator = (
        _progress_iter(B, progress_desc) if progress_desc is not None else range(B)
    )
    for b in iterator:
        out[b] = np.unravel_index(np.argmax(assignments[b]), assignments[b].shape)
    return out


def run_spanet(session, source, mask, progress_label=None):
    """Run a SPANet session on a batch.

    source : (N, J, 4) float32
    mask   : (N, J) bool  — True for valid jets
    Returns t1_idx, t2_idx each (N, 3) int — indices into the J-jet array.
    """
    print(f"      ONNX forward pass on {source.shape[0]:,} events...", flush=True)
    t_fwd = time.perf_counter()
    t1_assign, t2_assign, _, _ = session.run(
        None,
        {"source_data": source, "source_mask": mask},
    )
    print(
        f"      ONNX forward pass done in {_fmt_duration(time.perf_counter() - t_fwd)}",
        flush=True,
    )
    desc_t1 = f"{progress_label} argmax t1" if progress_label else None
    desc_t2 = f"{progress_label} argmax t2" if progress_label else None
    return (
        _extract_triplets(t1_assign, progress_desc=desc_t1),
        _extract_triplets(t2_assign, progress_desc=desc_t2),
    )


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


def run_comb_solver(session, model_input, progress_label=None):
    """Run a CombinatorialSolver session one event at a time (batch_size=1 baked in ONNX).

    model_input : (N, 7, 4) float32
    Returns t1_idx, t2_idx each (N, 3) int — indices into the 7-jet array.
    """
    N    = model_input.shape[0]
    best = np.empty(N, dtype=int)
    desc = (
        f"{progress_label} ONNX inference" if progress_label else None
    )
    iterator = _progress_iter(N, desc) if desc is not None else range(N)
    for i in iterator:
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
    # On a Condor slot the cgroup exposes only a subset of the node's cores, so
    # onnxruntime's default (one thread per logical core, pinned via
    # pthread_setaffinity_np) floods the log with affinity errors and
    # oversubscribes the slot. Pin the thread counts explicitly -- this both
    # silences the affinity calls and matches the requested CPUs. Honour
    # ORT_NUM_THREADS / OMP_NUM_THREADS if set, else default to 1 (the
    # comb_solver runs one event at a time, so intra-op threads buy little).
    n_threads = int(os.environ.get("ORT_NUM_THREADS")
                    or os.environ.get("OMP_NUM_THREADS")
                    or 1)
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = n_threads
    opts.inter_op_num_threads = n_threads
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return onnxruntime.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )


# ---------------------------------------------------------------------------
# Per-collection model dispatch
# ---------------------------------------------------------------------------

def _evaluate_collection(
    models, sessions, pt, eta, phi, e, px, py, pz, mask, candidate_infix,
):
    """Run every (model, session) pair on one jet collection.

    ``candidate_infix`` is inserted between ``{label}`` and ``Candidate_`` in the
    output branch names. Use ``""`` for reco (gives ``SPANetCandidate_pt`` etc.)
    and ``"Gen"`` for the GenJet pass (gives ``SPANetGenCandidate_pt`` etc.).
    Returns a dict of output branches.
    """
    n_chunk = pt.shape[0]

    comb_norm, comb_raw, _spher_7j, top7_idx = prepare_comb_input(
        pt, eta, phi, e, px, py, pz, mask
    )
    rows7    = np.arange(n_chunk)[:, None]
    top6_idx = top7_idx[:, :6]
    pt6  = pt [rows7, top6_idx]
    eta6 = eta[rows7, top6_idx]
    phi6 = phi[rows7, top6_idx]
    e6   = e  [rows7, top6_idx]
    px6  = px [rows7, top6_idx]
    py6  = py [rows7, top6_idx]
    pz6  = pz [rows7, top6_idx]
    # Use the real mask so SPANet ignores padded slots in low-multiplicity events
    # (e.g. GenJet). For reco, slimmer guarantees ≥6 jets so this is all-True.
    mask6 = mask[rows7, top6_idx]

    def _reg2(a):
        return ak.to_regular(ak.Array(a), axis=1)

    out = {}
    label_for_log = candidate_infix or "reco"

    for model_cfg, session in zip(models, sessions):
        prefix = _label_to_prefix(model_cfg["label"])
        mtype  = model_cfg["type"]
        plabel = f"{model_cfg['label']}/{label_for_log}"

        with _Timer(f"Running {model_cfg['label']} ({mtype}) on {label_for_log} jets ({n_chunk:,} events)"):
            if mtype == "spanet":
                if model_cfg["input_format"] == "cart":
                    source = np.stack([px6, py6, pz6, e6], axis=-1).astype(np.float32)
                else:
                    source = np.stack([pt6, eta6, phi6, e6], axis=-1).astype(np.float32)
                print(
                    f"      input shape {source.shape} ({source.dtype}), "
                    f"format={model_cfg['input_format']}",
                    flush=True,
                )
                t1_6, t2_6 = run_spanet(session, source, mask6, progress_label=plabel)
                t1_idx = top6_idx[rows7, t1_6]
                t2_idx = top6_idx[rows7, t2_6]
            elif mtype == "comb_solver":
                comb_in = comb_norm if model_cfg["normalized"] else comb_raw
                print(
                    f"      input shape {comb_in.shape} ({comb_in.dtype}), "
                    f"normalized={model_cfg['normalized']}",
                    flush=True,
                )
                t1_7, t2_7 = run_comb_solver(session, comb_in, progress_label=plabel)
                t1_idx = top7_idx[rows7, t1_7]
                t2_idx = top7_idx[rows7, t2_7]

            pt1, eta1, phi1, m1 = candidate_fourvec(pt, eta, phi, e, t1_idx)
            pt2, eta2, phi2, m2 = candidate_fourvec(pt, eta, phi, e, t2_idx)

        base = f"{prefix}{candidate_infix}Candidate_"
        out[f"{base}pt"]      = _reg2(np.stack([pt1,          pt2         ], axis=1).astype(np.float32))
        out[f"{base}eta"]     = _reg2(np.stack([eta1,         eta2        ], axis=1).astype(np.float32))
        out[f"{base}phi"]     = _reg2(np.stack([phi1,         phi2        ], axis=1).astype(np.float32))
        out[f"{base}mass"]    = _reg2(np.stack([m1,           m2          ], axis=1).astype(np.float32))
        out[f"{base}jetIdx0"] = _reg2(np.stack([t1_idx[:, 0], t2_idx[:, 0]], axis=1).astype(np.int32))
        out[f"{base}jetIdx1"] = _reg2(np.stack([t1_idx[:, 1], t2_idx[:, 1]], axis=1).astype(np.int32))
        out[f"{base}jetIdx2"] = _reg2(np.stack([t1_idx[:, 2], t2_idx[:, 2]], axis=1).astype(np.int32))

    return out


# ---------------------------------------------------------------------------
# Branch pass-through
# ---------------------------------------------------------------------------

def _passthrough_branches(chunk, regroup_collections):
    """Build an out_record dict from a chunk, regrouping any flat collection
    branches in ``regroup_collections`` into a single zipped record.

    For each collection name ``B`` in ``regroup_collections``:
      - if the chunk has a nested record ``B`` (with sub-fields), re-zip it so
        uproot emits one shared offset counter;
      - if the chunk has flat ``B_<field>`` branches, regroup them into a
        zipped ``B`` record (and drop the standalone ``nB`` counter, which
        uproot will regenerate).
    All other branches pass through unchanged.
    """
    chunk_fields = ak.fields(chunk)
    flat_by_collection = {
        b: [f for f in chunk_fields if f.startswith(f"{b}_")]
        for b in regroup_collections
    }
    all_flat       = {f for fs in flat_by_collection.values() for f in fs}
    skip_counters  = {f"n{b}" for b, fs in flat_by_collection.items() if fs}
    nested_targets = set(regroup_collections)

    out_record = {}
    for branch in chunk_fields:
        val = chunk[branch]
        if branch in nested_targets and ak.fields(val):
            out_record[branch] = ak.zip({f: val[f] for f in ak.fields(val)})
        elif branch in skip_counters or branch in all_flat:
            continue
        else:
            out_record[branch] = val

    for b, fs in flat_by_collection.items():
        if fs:
            out_record[b] = ak.zip({f[len(b) + 1:]: chunk[f] for f in fs})

    return out_record


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(input_path, output_path, config, config_path, in_tree_name, chunk_size,
             run_on_genjets=False):
    job_t0 = time.perf_counter()
    models  = config["models"]
    version = config.get("metadata", {}).get("version", "unknown")

    try:
        in_size = os.path.getsize(input_path)
        in_size_str = f"{in_size / (1024**2):.1f} MiB"
    except OSError:
        in_size_str = "?"

    print(f"Job started:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input:        {input_path}  (tree: {in_tree_name}, {in_size_str})")
    print(f"Output:       {output_path}")
    print(f"Config:       {config_path}")
    print(f"Version:      {version}")
    print(f"Chunk size:   {chunk_size:,}")
    print(f"Models:       {[m['label'] for m in models]}")
    print(f"Run on GenJet: {run_on_genjets}")
    print(f"onnxruntime providers (available): {onnxruntime.get_available_providers()}")
    print()

    print("Loading ONNX sessions...")
    sessions  = []
    load_t0 = time.perf_counter()
    for m in models:
        with _Timer(f"{m['label']} ({m['type']})  <-  {m['path']}"):
            sess = _load_session(m["path"])
            print(
                f"    providers used: {sess.get_providers()}",
                flush=True,
            )
            sessions.append(sess)
    print(f"  All {len(sessions)} session(s) loaded in "
          f"{_fmt_duration(time.perf_counter() - load_t0)}.")
    print()

    with uproot.open(input_path) as in_file:
        if in_tree_name not in in_file:
            sys.exit(
                f"Tree '{in_tree_name}' not found in {input_path}. "
                f"Available keys: {list(in_file.keys())}"
            )

        tree      = in_file[in_tree_name]
        tree_keys = set(tree.keys())

        # Jet kinematics may be stored either as a nested ScoutingPFJet record
        # ("ScoutingPFJet.pt", …) or as flat NanoAOD-style branches
        # ("ScoutingPFJet_pt", …). Accept whichever the input uses.
        fmt = jet_format(tree_keys)
        if fmt is None:
            sys.exit(
                f"Expected jet branch '{_JET_BRANCH}.pt' (nested) or "
                f"'{_JET_BRANCH}_pt' (flat) not found in tree "
                f"'{in_tree_name}'. Available keys: {sorted(tree_keys)[:20]}"
            )
        print(f"Jet layout: {fmt} ('{_JET_BRANCH}{'.' if fmt == 'nested' else '_'}pt')")

        gen_fmt = jet_format(tree_keys, branch=_GEN_JET_BRANCH)
        if gen_fmt is not None:
            print(f"GenJet layout: {gen_fmt} ('{_GEN_JET_BRANCH}{'.' if gen_fmt == 'nested' else '_'}pt') — will pass through")
        if run_on_genjets and gen_fmt is None:
            sys.exit(
                f"--run-on-genjets was requested but no '{_GEN_JET_BRANCH}' "
                f"collection was found in tree '{in_tree_name}'. This input "
                "does not look like MC."
            )
        if run_on_genjets:
            print(f"Will also run all models on the {_GEN_JET_BRANCH} collection.")

        total_entries = tree.num_entries
        n_chunks      = max(1, (total_entries + chunk_size - 1) // chunk_size)
        print(f"Tree has {total_entries:,} events -> {n_chunks} chunk(s) of up to {chunk_size:,}")
        print()

        total_in = total_out = 0
        chunk_times = []

        with uproot.recreate(output_path) as out_file:
            out_tree  = None
            chunk_num = 0

            for chunk in tree.iterate(library="ak", step_size=chunk_size):
                chunk_t0  = time.perf_counter()
                chunk_num += 1
                n_chunk    = len(chunk)
                total_in  += n_chunk
                print(
                    f"[{chunk_num}/{n_chunks}] Chunk of {n_chunk:,} events"
                    f"  (running total: {total_in:,}/{total_entries:,},"
                    f" {100.0 * total_in / total_entries:.1f}%)"
                )

                # Pad to ≥7 slots so CombSolver always sees (N, 7, 4); reco
                # already has ≥6 jets thanks to the slimmer, but GenJets may
                # have fewer in some events.
                with _Timer("Converting reco branches to numpy arrays"):
                    pt, eta, phi, px, py, pz, e, mask = chunk_to_numpy(chunk, min_jets=7)
                    print(
                        f"    padded shape: {pt.shape}, "
                        f"valid jets/event mean={mask.sum(1).mean():.2f} "
                        f"min={mask.sum(1).min()} max={mask.sum(1).max()}",
                        flush=True,
                    )

                # Pass through all original top-level branches, regrouping any
                # flat ScoutingPFJet_*/GenJet_*/GenPart_* layout into a single
                # nested record so the output is layout-independent.
                with _Timer("Copying input branches"):
                    out_record = _passthrough_branches(
                        chunk, [_JET_BRANCH, _GEN_JET_BRANCH, _GEN_PART_BRANCH]
                    )
                    print(f"    {len(out_record)} input branches passed through", flush=True)

                out_record.update(_evaluate_collection(
                    models, sessions, pt, eta, phi, e, px, py, pz, mask,
                    candidate_infix="",
                ))

                if run_on_genjets:
                    with _Timer("Converting GenJet branches to numpy arrays"):
                        g_pt, g_eta, g_phi, g_px, g_py, g_pz, g_e, g_mask = chunk_to_numpy(
                            chunk, branch=_GEN_JET_BRANCH, min_jets=7,
                        )
                        print(
                            f"    padded shape: {g_pt.shape}, "
                            f"valid jets/event mean={g_mask.sum(1).mean():.2f} "
                            f"min={g_mask.sum(1).min()} max={g_mask.sum(1).max()}",
                            flush=True,
                        )
                    out_record.update(_evaluate_collection(
                        models, sessions,
                        g_pt, g_eta, g_phi, g_e, g_px, g_py, g_pz, g_mask,
                        candidate_infix="Gen",
                    ))

                total_out += n_chunk

                with _Timer(f"Writing {n_chunk:,} events ({len(out_record)} branches) to output"):
                    if out_tree is None:
                        out_file.mktree(
                            "events",
                            {name: arr.type for name, arr in out_record.items()},
                        )
                        out_tree = out_file["events"]
                    out_tree.extend(out_record)

                chunk_dt = time.perf_counter() - chunk_t0
                chunk_times.append(chunk_dt)
                rate     = n_chunk / chunk_dt if chunk_dt > 0 else 0.0
                # Project remaining time from the average chunk rate so far.
                remaining = total_entries - total_in
                avg_rate  = total_in / sum(chunk_times) if sum(chunk_times) > 0 else rate
                eta_s     = remaining / avg_rate if avg_rate > 0 else 0.0
                print(
                    f"  Chunk {chunk_num}/{n_chunks} complete in "
                    f"{_fmt_duration(chunk_dt)} ({rate:.1f} events/s). "
                    f"Job ETA: {_fmt_duration(eta_s)}"
                )
                print()

            # Version histogram (StrCategory is the only reliable way to store
            # strings in uproot; byte-string TTree branches trigger RNTuple routing).
            version_hist = bh.Histogram(bh.axis.StrCategory([version]), storage=bh.storage.Double())
            version_hist.view()[0] = 1.0
            out_file["version"] = version_hist

            # Pass through cutflow histogram if present
            if "cutflow" in in_file:
                with _Timer("Copying cutflow histogram from input", indent=""):
                    out_file["cutflow"] = in_file["cutflow"]

    wall = time.perf_counter() - job_t0
    avg_rate = total_in / wall if wall > 0 else 0.0
    try:
        out_size = os.path.getsize(output_path)
        out_size_str = f"{out_size / (1024**2):.1f} MiB"
    except OSError:
        out_size_str = "?"

    print("=" * 72)
    print(f"Done.  {total_in:,} events processed -> {total_out:,} written.")
    print(f"Output file:    {output_path} ({out_size_str})")
    if chunk_times:
        print(
            f"Chunks:         {len(chunk_times)} "
            f"(min {_fmt_duration(min(chunk_times))}, "
            f"max {_fmt_duration(max(chunk_times))}, "
            f"avg {_fmt_duration(sum(chunk_times) / len(chunk_times))})"
        )
    print(f"Average rate:   {avg_rate:.1f} events/s")
    print(f"Job finished:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total wall time: {_fmt_duration(wall)}")
    print("=" * 72)


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
    parser.add_argument(
        "--run-on-genjets", action="store_true",
        help=(
            "Also run every model on the GenJet collection (requires MC input "
            "with a GenJet branch). Emits additional {label}GenCandidate_* "
            "branches whose jet indices point into the GenJet arrays."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main_t0 = time.perf_counter()
    try:
        evaluate(
            input_path=args.input,
            output_path=args.output,
            config=cfg,
            config_path=args.config,
            in_tree_name=args.tree,
            chunk_size=args.chunk_size,
            run_on_genjets=args.run_on_genjets,
        )
    except BaseException as exc:
        wall = time.perf_counter() - main_t0
        print(
            f"\nJob aborted after {_fmt_duration(wall)}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
