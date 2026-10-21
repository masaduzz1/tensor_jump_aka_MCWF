"""Overlay TJM runs that differ only in the Trotter step dt (no exact data needed, e.g. L=40).

Panels: (a) <X_iX_{i+r}>, (b) <Z_iZ_{i+r}> (r >= 1), (c) S^XX(q); +-1 sem bands, open markers at true coordinates.
Refuses to overwrite an existing figure.

Usage:
    python plot_dt_compare.py --L 40 --out fig/dtcompare_<...>.pdf "dt=0.05:data/tjm_...csv" "dt=0.1:data/tjm_...csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_exact_tjm import HERE, corr_matrix, corr_vs_r, read_row
from structure_factor import structure_factor_pbc

MARKERS = "os^"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label:path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--L", type=int, default=40)
    a = ap.parse_args()
    L, out = a.L, HERE / a.out
    if out.exists() or out.with_suffix(".png").exists():
        raise SystemExit(f"{out.name} exists; not overwriting")
    r = np.arange(L // 2 + 1)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    for k, s in enumerate(a.runs):
        lab, path = s.split(":", 1)
        path = HERE / path
        t = read_row(path)
        e = read_row(path.with_name(path.name.replace("tjm_", "tjm-err_", 1)))
        mk, col = MARKERS[k % len(MARKERS)], f"C{k}"
        zavg = np.mean([t[f"Z{i}"] for i in range(L)])
        for i, (p, r0) in enumerate([("X", 0), ("Z", 1)]):
            c = corr_vs_r(corr_matrix(t, p, L))[: len(r)]
            sm = np.array([0.0] + [e[f"{p}0{p}{j}"] for j in r[1:]])
            ax[i].fill_between(r[r0:], (c - sm)[r0:], (c + sm)[r0:], color=col, alpha=0.18, lw=0)
            ax[i].plot(r[r0:], c[r0:], mk, color=col, mfc="none", mew=1.2, ms=5,
                       label=lab + (rf", $\langle Z\rangle$={zavg:.3f}" if p == "Z" else ""))
        ax[1].axhline(zavg**2, color=col, ls=":", lw=1)
        q, S = structure_factor_pbc(corr_matrix(t, "X", L))
        dS = np.sqrt(sum(e[f"X0X{j}"] ** 2 for j in range(1, L)))
        ax[2].fill_between(q / np.pi, S - dS, S + dS, color=col, alpha=0.18, lw=0)
        ax[2].plot(q / np.pi, S, mk + "-", color=col, mfc="none", mew=1.2, ms=4, lw=0.8, label=lab)
    ax[0].axhline(0, color="gray", lw=0.5)
    ax[0].set(xlabel="$r$", ylabel=r"$\langle X_iX_{i+r}\rangle$")
    ax[1].set(xlabel="$r$", ylabel=r"$\langle Z_iZ_{i+r}\rangle$ (dotted: $\langle Z\rangle^2$)")
    ax[2].set(xlabel=r"$q/\pi$", ylabel=r"$S^{XX}(q)$", xlim=(-1, 1))
    for x, lab in zip(ax, "abc"):
        x.legend(fontsize=8)
        x.text(0.02, 0.96, f"({lab})", transform=x.transAxes, va="top", fontweight="bold")
    if a.title:
        fig.suptitle(a.title)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"saved {out.relative_to(HERE)} (+ .png)")


if __name__ == "__main__":
    main()
