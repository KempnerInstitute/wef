"""
Thin wrapper: runs the 2f1p multispec analysis for the k0 (knollen-off) conditions.
Spec keys are 2f1p_k0_{AltB,AeqB,AgtB,control_a,control_b}.

Usage:
    python analysis_2f1p_k0_multispec.py --evals_dir <run_dir>/evals [--out_dir ...]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from analysis_2f1p_multispec import run as _run


def run(evals_dir, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(evals_dir, "..", "multi_eval", "2f1p_k0")
    _run(evals_dir, out_dir=out_dir, spec_prefix="2f1p_k0")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args(argv)
    run(args.evals_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
