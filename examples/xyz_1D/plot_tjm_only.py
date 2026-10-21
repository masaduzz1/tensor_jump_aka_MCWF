"""Plot TJM steady-state correlators for runs without exact data (e.g. L=40).

Panels: (a) <X_iX_{i+r}>, (b) <Z_iZ_{i+r}> (r >= 1), (c) S^XX(q); +-1 sem bands, open markers.
Saves fig/tjm_only_<stem>.{pdf,png}; refuses to overwrite an existing figure.

Usage:
    python plot_tjm_only.py data/tjm_<stem>.csv [--L 40]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--L", type=int, default=40)
    a = ap.parse_args()
    L, path = a.L, HERE / a.csv
    stem = path.stem.removeprefix("tjm_")
    out = HERE / "fig" / f"tjm_only_{stem}.pdf"
    if out.exists() or out.with_suffix(".png").exists():
        raise SystemExit(f"{out.name} exists; not overwriting")
    t = read_row(path)
    e = read_row(path.with_name(path.name.replace("tjm_", "tjm-err_", 1)))
    r = np.arange(L // 2 + 1)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    for i, (p, r0, mk) in enumerate([("X", 0, "o"), ("Z", 1, "s")]):
        c = corr_vs_r(corr_matrix(t, p, L))[: len(r)]
        s = np.array([0.0] + [e[f"{p}0{p}{k}"] for k in r[1:]])
        ax[i].fill_between(r[r0:], (c - s)[r0:], (c + s)[r0:], color="C0", alpha=0.2, lw=0)
        ax[i].plot(r[r0:], c[r0:], mk, color="C0", mfc="none", mew=1.2, ms=5, label="TJM")
    zavg = np.mean([t[f"Z{i}"] for i in range(L)])
    ax[1].axhline(zavg**2, color="gray", ls=":", lw=1, label=r"$\langle Z\rangle^2$")
    q, S = structure_factor_pbc(corr_matrix(t, "X", L))
    sX = np.array([e[f"X0X{k}"] for k in range(1, L)])
    dS = np.sqrt(np.sum(sX**2))
    ax[2].fill_between(q / np.pi, S - dS, S + dS, color="C0", alpha=0.2, lw=0)
    ax[2].plot(q / np.pi, S, "^-", color="C0", mfc="none", mew=1.2, ms=4, lw=0.8, label="TJM")
    ax[0].axhline(0, color="gray", lw=0.5)
    ax[0].set(xlabel="$r$", ylabel=r"$\langle X_iX_{i+r}\rangle$")
    ax[1].set(xlabel="$r$", ylabel=r"$\langle Z_iZ_{i+r}\rangle$")
    ax[2].set(xlabel=r"$q/\pi$", ylabel=r"$S^{XX}(q)$", xlim=(-1, 1))
    for x, lab in zip(ax, "abc"):
        x.legend(fontsize=8)
        x.text(0.02, 0.96, f"({lab})", transform=x.transAxes, va="top", fontweight="bold")
    fig.suptitle(stem.replace("_", " ") + f"   (no exact data; <Z>={zavg:.3f})", fontsize=9)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"saved {out.relative_to(HERE)} (+ .png)")


if __name__ == "__main__":
    main()
