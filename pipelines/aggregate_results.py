#!/usr/bin/env python3
"""Result aggregation"""

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

REF_BIN, REF_FUNC = "1" * 15, "0.5"
DS_DL_ALL = "mimic_extract_complex_func_stepwise_0_04_05_06_1_all"
DS_PEQ_ALL = "mimic_extract_complex_func_stepwise_04_05_06_all"
DS_PEQ_051_ALL = "mimic_extract_complex_func_stepwise_0_05_1_all"
DL = "dltmle_correct_func"
PEQ_046 = "peq_net_func (joint: {0.4, 0.5, 0.6})"
PEQ_00461 = "peq_net_func (joint: {0, 0.4, 0.5, 0.6, 1})"
PEQ_051 = "peq_net_func (joint: {0, 0.5, 1})"
W = (44, 46, 22)


def fkey(x):
    try:
        return 0, float(x)
    except (TypeError, ValueError):
        return 1, x


def metric_prefix(run_dir):
    return next((p + "_" for p in ("gcomp", "ltmle", "deepace") if run_dir.startswith(p)), "ltmle_")


def load_one(root, method, run_dir, ds, seeds, norm=lambda s: s):
    rows, missing, est_key = [], [], metric_prefix(run_dir) + "estimate"
    for seed in seeds:
        folder = root / run_dir / f"exp_seed={seed}"
        matches = sorted(folder.glob(f"*_{ds}_capo_results.json")) if folder.is_dir() else []
        if not matches:
            missing.append(seed)
            continue
        if len(matches) > 1:
            print(f"Warning: multiple CAPO files for {run_dir} {ds} seed={seed}; using {matches[0]}", file=sys.stderr)
        with matches[0].open() as f:
            data = json.load(f)
        rows += [
            {
                "method": method,
                "dataset": ds,
                "sequence": norm(seq),
                "seed": seed,
                "ground_truth": float(r["ground_truth"]),
                "estimate": float(r[est_key]),
            }
            for seq, r in data["results"].items()
        ]
    if not rows:
        return rows, [f"no CAPO JSON for {run_dir} / {ds} (looked for seeds={list(seeds)})"]
    return rows, ([f"{run_dir} / {ds}: incomplete seeds {missing}"] if missing else [])


def load_specs(root, specs, seeds, norm=lambda s: s):
    rows, warns = [], []
    for spec in specs:
        r, w = load_one(root, *spec, seeds, norm)
        rows += r
        warns += w
    return rows, warns


def cate(rows, ref_seq):
    grouped, out = {}, []
    for r in rows:
        grouped.setdefault((r["method"], r["dataset"], r["seed"]), {})[r["sequence"]] = r
    for (method, ds, seed), by_seq in grouped.items():
        if ref_seq not in by_seq:
            continue
        ref = by_seq[ref_seq]
        for cf, raw in by_seq.items():
            if cf == ref_seq:
                continue
            bias = (raw["estimate"] - ref["estimate"]) - (raw["ground_truth"] - ref["ground_truth"])
            out.append({"method": method, "dataset": ds, "cf": cf, "seed": seed, "abs_bias_cate": abs(bias), "bias_cate": bias})
    return out


def cate_specs(root, specs, seeds, keep):
    rows, warns = load_specs(root, specs, seeds)
    return [r for r in cate(rows, REF_FUNC) if float(r["cf"]) in keep], warns


def section3_order(setups):
    setup_set, ordered = set(setups), []
    for cf in sorted({cf for ds, cf in setup_set if ds in (DS_PEQ_ALL, DS_DL_ALL)}, key=fkey):
        ordered += [p for p in ((DS_PEQ_ALL, cf), (DS_DL_ALL, cf)) if p in setup_set]
    return ordered + sorted(s for s in setup_set if s not in ordered)


def table(title, rows, n_seeds, *, setup_order=None, by_cf=False, method_order=()):
    print("\n" + "=" * 92)
    if not rows:
        print(title)
        print("=" * 92)
        print("(no rows - check that result JSONs exist for the requested seeds)")
        return []
    stats = {}
    for ds, cf, method in {(r["dataset"], r["cf"], r["method"]) for r in rows}:
        sub = [r for r in rows if (r["dataset"], r["cf"], r["method"]) == (ds, cf, method)]
        abs_vals, bias_vals = [r["abs_bias_cate"] for r in sub], [r["bias_cate"] for r in sub]
        stats[(ds, cf, method)] = (
            st.mean(abs_vals),
            st.stdev(abs_vals) if len(abs_vals) > 1 else 0.0,
            math.sqrt(st.mean(b * b for b in bias_vals)),
            len(sub),
        )
    counts, warns = sorted({v[3] for v in stats.values()}), []
    if len(counts) > 1:
        title += " (replicate counts vary across rows - see warnings)"
        warns.append(f'Table "{title[:72]}": inconsistent number of seeds across rows - counts {counts}')
    else:
        title += f" (n={counts[0]})"
        if counts[0] != n_seeds:
            warns.append(f'Table "{title[:72]}": expected {n_seeds} seeds but each row used n={counts[0]}')
    rank = {m: i for i, m in enumerate(method_order)}
    print(title)
    print("=" * 92)
    print(f"  {'method':<{W[0]}}  {'dataset':<{W[1]}}  {'CF':<{W[2]}}  {'|bias(CATE)| mean±std':>26}  {'RMSE(bias CATE)':>14}")
    if by_cf:
        blocks = [(None, cf) for cf in sorted({r["cf"] for r in rows}, key=fkey)]
    else:
        setups = {(r["dataset"], r["cf"]) for r in rows}
        blocks = section3_order(setups) if setup_order == "section3" else sorted(setups)
    for ds, cf in blocks:
        methods = {r["method"] for r in rows if r["cf"] == cf and (ds is None or r["dataset"] == ds)}
        for method in sorted(methods, key=lambda m: (rank.get(m, 9999), m)):
            row_ds = ds or next(r["dataset"] for r in rows if r["cf"] == cf and r["method"] == method)
            mean, sd, rmse, _ = stats[(row_ds, cf, method)]
            print(f"  {method:<{W[0]}}  {row_ds:<{W[1]}}  {f'CF={cf}':<{W[2]}}  {f'{mean:.4f} ± {sd:.4f}':>26}  {rmse:>14.4f}")
        print("-----")
    return warns


def add(warns, title, rows, seeds, **kwargs):
    warns += table(title, rows, len(seeds), **kwargs)


def nz(ds):
    return 5 if ds.startswith("mimic_extract_complex") else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None, help="Path to results/ (default: <repo>/results)")
    parser.add_argument("--n-repeats", type=int, default=20, help="Number of consecutive seeds to aggregate (default: 20)")
    parser.add_argument("--exp-seed-start", type=int, default=1700, help="First seed when using --n-repeats (default: 1700)")
    args = parser.parse_args()
    root = args.results_dir.expanduser().resolve() if args.results_dir else Path(__file__).resolve().parents[1] / "results"
    if not root.is_dir():
        sys.exit(f"Results directory not found: {root}")
    
    seeds = list(range(args.exp_seed_start, args.exp_seed_start + args.n_repeats))
    print(f"Seeds: {seeds}")
    warns = []

    det_methods = [("gcomp", "gcomp"), ("ltmle", "ltmle"), ("deepace", "deepace_tuned_numz{nz}"), ("dltmle_correct_01", "dltmle_correct_01_tuned_numz{nz}"), ("peq_net", "peq_net_01_tuned_numz{nz}")]
    specs = [(m, tmpl.format(nz=nz(ds)) if "{nz}" in tmpl else tmpl, ds) for m, tmpl in det_methods for ds in ("mimic_extract", "mimic_extract_complex")]
    rows, w = load_specs(root, specs, seeds, lambda s: s.replace(" ", "").replace(",", "").replace("[", "").replace("]", ""))
    warns += w
    add(warns, f"1) CATE error (|bias|), deterministic binary sequences (reference = all-ones `{REF_BIN}`)", cate(rows, REF_BIN), seeds)

    func_methods = [(DL, "dltmle_correct_func_tuned_numz{nz}"), ("peq_net_func", "peq_net_func_tuned_numz{nz}")]
    specs = [(m, tmpl.format(nz=nz(ds)), ds) for m, tmpl in func_methods for ds in ("mimic_extract_complex_func_stepwise", "mimic_extract_func_stepwise")]
    rows, w = load_specs(root, specs, seeds)
    warns += w
    add(warns, f"2) CATE error (|bias|), functional stepwise - overlap differs only in first two steps (reference = `{REF_FUNC}`)", cate(rows, REF_FUNC), seeds)

    specs = [(DL, "dltmle_correct_func_tuned_numz5", DS_DL_ALL), ("peq_net_func", "peq_net_func_tuned_numz5", DS_PEQ_ALL), (DL, "dltmle_correct_func_tuned_numz0", "mimic_extract_func_stepwise_04_05_06_all"), ("peq_net_func", "peq_net_func_tuned_numz0", "mimic_extract_func_stepwise_04_05_06_all")]
    rows, w = load_specs(root, specs, seeds)
    warns += w
    rows = [r for r in cate(rows, REF_FUNC) if not (r["dataset"] == DS_DL_ALL and r["method"] == DL and float(r["cf"]) in {0.0, 1.0})]
    add(warns, f"3) CATE error (|bias|), functional stepwise - overlap differs at all steps (reference = `{REF_FUNC}`)", rows, seeds, setup_order="section3")

    specs = [("dltmle_correct_func (default trainer)", "dltmle_correct_func_tuned_numz5", "mimic_extract_complex_func_stepwise"), ("dltmle_correct_func (finetuning)", "dltmle_correct_finetune_func_numz5", "mimic_extract_complex_func_stepwise"), ("dltmle_correct_func (multi-head Q)", "dltmle_correct_multiQhead_func_numz5", "mimic_extract_complex_func_stepwise"), ("peq_net_func", "peq_net_func_tuned_numz5", "mimic_extract_complex_func_stepwise")]
    rows, w = load_specs(root, specs, seeds)
    warns += w
    add(warns, "4) Ablation - functional early overlap only (`mimic_extract_complex_func_stepwise`): DL-TMLE default vs finetuning vs multi-head Q; PEQ-Net", cate(rows, REF_FUNC), seeds)

    extreme1 = [(DL, "dltmle_correct_func_tuned_numz5", DS_DL_ALL), (PEQ_046, "peq_net_func_tuned_numz5", DS_PEQ_ALL), (PEQ_00461, "peq_net_func_tuned_numz5", DS_DL_ALL)]
    rows, w = cate_specs(root, extreme1, seeds, {0.4, 0.6})
    warns += w
    add(warns, f"5) Ablation - extreme thresholds & PEQ joint-training - Part 1 - CF in {{0.4, 0.6}} (reference = `{REF_FUNC}`)", rows, seeds, by_cf=True, method_order=[DL, PEQ_046, PEQ_00461])

    extreme2 = [(DL, "dltmle_correct_func_tuned_numz5", DS_DL_ALL), (PEQ_051, "peq_net_func_tuned_numz5", DS_PEQ_051_ALL), (PEQ_00461, "peq_net_func_tuned_numz5", DS_DL_ALL)]
    rows, w = cate_specs(root, extreme2, seeds, {0.0, 1.0})
    warns += w
    add(warns, f"5) Ablation - extreme thresholds & PEQ joint-training - Part 2 - CF in {{0, 1}} (reference = `{REF_FUNC}`)", rows, seeds, by_cf=True, method_order=[DL, PEQ_051, PEQ_00461])

    if warns := sorted(set(warns)):
        print("\n" + "-" * 92)
        print(f"Warnings ({len(warns)} unique):")
        for line in warns[:50]:
            print(f"  - {line}")
        if len(warns) > 50:
            print(f"  ... ({len(warns) - 50} more)")


if __name__ == "__main__":
    main()
