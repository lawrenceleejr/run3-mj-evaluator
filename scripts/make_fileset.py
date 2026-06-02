#!/usr/bin/env python3
"""make_fileset.py - Convert plain-text file lists to a coffea-style fileset JSON.

Each positional argument is a .txt file whose lines are bare EOS paths
(/store/...).  The dataset name is derived from the filename by stripping a
trailing '_filelist' suffix and the extension.  Paths are prefixed with the
XRootD redirector so condor jobs can stream files over WAN.

For the evaluator the inputs are *slimmed* files (tree 'events'), so the
default recorded tree name is 'events'.

Usage:
    python make_fileset.py filelists/*.txt -o fileset.json
    python make_fileset.py sample_a.txt sample_b.txt -o fileset.json \\
        --redirector root://cmsxrootd.fnal.gov/
"""

import argparse
import json
import os
import sys


# Default redirector: LPC EOS. The evaluator's inputs are the slimmer's
# outputs, which live in personal/group EOS (/store/group/..., /store/user/...)
# and must be read through the EOS redirector. XCache (root://xcache/) only
# serves the globally-distributed CMS namespace and is not resolvable for these
# files -- it yields "[FATAL] Invalid address". Use root://cmsxrootd.fnal.gov/
# for global CMS access if you ever point this at official datasets.
DEFAULT_REDIRECTOR = "root://cmseos.fnal.gov/"


def dataset_name(txt_path: str) -> str:
    """Derive a dataset label from a filelist filename."""
    stem = os.path.splitext(os.path.basename(txt_path))[0]
    for suffix in ("_filelist", "_files", "_file_list"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def read_paths(txt_path: str, redirector: str, tree: str) -> dict:
    """Read a filelist and return {xrd_path: tree} dict."""
    files = {}
    with open(txt_path) as f:
        for line in f:
            path = line.strip()
            if not path or path.startswith("#"):
                continue
            if path.startswith("root://"):
                xrd = path          # already has a redirector
            elif path.startswith("/store/"):
                xrd = redirector + "/" + path.lstrip("/")
            else:
                print(f"  Warning: unexpected path format, skipping: {path}", file=sys.stderr)
                continue
            files[xrd] = tree
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Build a coffea-style fileset JSON from plain-text file lists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs", nargs="+", metavar="FILELIST",
        help="One or more .txt files, one /store/... path per line",
    )
    parser.add_argument(
        "-o", "--output", default="fileset.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--redirector", default=DEFAULT_REDIRECTOR,
        help="XRootD redirector prefix",
    )
    parser.add_argument(
        "--tree", default="events",
        help="Tree name to record for every file",
    )
    args = parser.parse_args()

    fileset = {}
    for txt in args.inputs:
        name  = dataset_name(txt)
        files = read_paths(txt, args.redirector, args.tree)
        if not files:
            print(f"Warning: no valid paths found in {txt}", file=sys.stderr)
            continue
        fileset[name] = {"files": files}
        print(f"  {name}: {len(files)} files")

    with open(args.output, "w") as f:
        json.dump(fileset, f, indent=4)

    total = sum(len(v["files"]) for v in fileset.values())
    print(f"\nWrote {len(fileset)} datasets ({total} files total) → {args.output}")


if __name__ == "__main__":
    main()
