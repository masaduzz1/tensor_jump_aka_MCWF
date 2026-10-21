"""Compare several TJM runs (e.g. bond-dimension or trajectory scans) with the exact steady state.

Panels: (a) <X_iX_{i+r}>, (b) <Z_iZ_{i+r}> (r >= 1), (c) S^XX(q). Each TJM run is drawn with open markers
at its true coordinates and a shaded +-1 standard-error band; the exact data is dashed. Also prints, per run,
the max deviation from exact and from a reference run, and the mean max bond dimension reached.

Usage:
    python compare_runs.py --out fig/scan_chi.pdf --title "..." \
        "chi=8:data/tjm_..._chi8.csv" "chi=16:data/tjm_..._chi16.csv" ...
The first run listed after --ref (if given) is the reference for the "vs ref" column.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_exact_tjm import HERE, corr_matrix, corr_vs_r, exact_file, params_from_name, read_row
from structure_factor import structure_factor_pbc

MARKERS = "osv^Dp<>"


def load(path, L):
    path = Path(path)
    t = read_row(path)
    e = read_row(path.with_name(path.name.replace("tjm_", "tjm-err_", 1)))
    out = {}
    for p in "XZ":
        c = corr_vs_r(corr_matrix(t, p, L))
        sem = np.array([0.0] + [e[f"{p}0{p}{r}"] for r in range(1, L)])
        out[p] = (c, sem)
    diag = path.with_name(path.name.replace("tjm_", "diag_", 1).replace(".csv", ".npz"))
    out["chi"] = float(np.load(diag)["max_bond"][-1]) if diag.exists() else np.nan
    out["Zavg"] = np.mean([t[f"Z{i}"] for i in range(L)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label:path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--ref", default=None, help="label of the reference run")
    ap.add_argument("--L", type=int, default=12)
    a = ap.parse_args()
    L = a.L
    runs = [(s.split(":", 1)[0], s.split(":", 1)[1]) for s in a.runs]
    data = {lab: load(HERE / p, L) for lab, p in runs}
    ex = read_row(exact_file(params_from_name(Path(runs[0][1]))))
    ce = {p: corr_vs_r(corr_matrix(ex, p, L)) for p in "XZ"}

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    r = np.arange(L)
    for k, (lab, _) in enumerate(runs):
        d, mk, col = data[lab], MARKERS[k % len(MARKERS)], f"C{k}"
        for i, (p, r0) in enumerate([("X", 0), ("Z", 1)]):
            c, s = d[p]
            ax[i].fill_between(r[r0:], (c - s)[r0:], (c + s)[r0:], color=col, alpha=0.18, lw=0)
            ax[i].plot(r[r0:], c[r0:], mk, color=col, mfc="none", mew=1.2, ms=6, label=lab)
        # S(q) band: sem propagated assuming independent C(r) errors
        q, S = structure_factor_pbc(corr_matrix(read_row(HERE / dict(runs)[lab]), "X", L))
        sX = d["X"][1]
        dS = np.sqrt(np.sum(sX**2))
        ax[2].fill_between(q / np.pi, S - dS, S + dS, color=col, alpha=0.18, lw=0)
        ax[2].plot(q / np.pi, S, mk + "-", color=col, mfc="none", mew=1.2, ms=6, lw=0.8, label=lab)
    ax[0].plot(r, ce["X"], "k--", lw=1.3, label="exact")
    ax[1].plot(r[1:], ce["Z"][1:], "k--", lw=1.3, label="exact")
    q, S = structure_factor_pbc(corr_matrix(ex, "X", L))
    ax[2].plot(q / np.pi, S, "k--", lw=1.3, label="exact")
    ax[0].axhline(0, color="gray", lw=0.5)
    ax[0].set(xlabel="$r$", ylabel=r"$\langle X_iX_{i+r}\rangle$")
    ax[1].set(xlabel="$r$", ylabel=r"$\langle Z_iZ_{i+r}\rangle$")
    ax[2].set(xlabel=r"$q/\pi$", ylabel=r"$S^{XX}(q)$", xlim=(-1, 1))
    for x, lab in zip(ax, "abc"):
        x.legend(fontsize=8)
        x.text(0.02, 0.96, f"({lab})", transform=x.transAxes, va="top", fontweight="bold")
    if a.title:
        fig.suptitle(a.title)
    out = HERE / a.out
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"saved {out} (+ .png)")

    ref = data.get(a.ref)
    print(f"{'run':>10} {'chi_max':>8} {'max|dXX| ex':>12} {'max|dZZ| ex':>12} {'max|dXX| ref':>13} {'max|dZZ| ref':>13} {'sem XX':>7} {'<Z>':>8}")
    for lab, _ in runs:
        d = data[lab]
        dx = np.abs(d["X"][0][1:] - ce["X"][1:]).max()
        dz = np.abs(d["Z"][0][1:] - ce["Z"][1:]).max()
        rx = np.abs(d["X"][0][1:] - ref["X"][0][1:]).max() if ref else np.nan
        rz = np.abs(d["Z"][0][1:] - ref["Z"][0][1:]).max() if ref else np.nan
        print(f"{lab:>10} {d['chi']:>8.1f} {dx:>12.4f} {dz:>12.4f} {rx:>13.4f} {rz:>13.4f} {d['X'][1][1:].mean():>7.4f} {d["Zavg"]:>8.4f}")
    print(f"{'exact':>10} {'':>8} {'':>12} {'':>12} {'':>13} {'':>13} {'':>7} {np.mean([ex[f'Z{i}Z{i}'] for i in range(L)]):>8.4f}")


if __name__ == "__main__":
    main()
