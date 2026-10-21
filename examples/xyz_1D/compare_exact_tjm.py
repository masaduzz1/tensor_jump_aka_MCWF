"""Compare steady-state correlators from tensor jump (data/tjm_*.csv) with exact L=12 data.

For every TJM file in data/, finds the matching exact file (last row = steady state)
and saves fig/compare_<stem>_<P><P>.png for each correlator P in {X, Z} present:
  left/middle: TJM and exact <P_i P_j> heatmaps; right: translation-averaged C(r).

Usage:
    conda activate tensorjump
    python compare_exact_tjm.py            # all files in data/
    python compare_exact_tjm.py data/tjm_Jx3_Jy2_....csv
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EXACT_DIR = HERE.parent / "data-dd-heisenberg" / "data-dd-heisenberg" / "data-exact"


def read_row(path, last=True):
    header = open(path).readline().strip().split(",")
    d = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
    return dict(zip(header, d[-1] if last else d[0]))


def corr_matrix(row, p, L):
    C = np.eye(L)
    for i in range(L):
        for j in range(i + 1, L):
            C[i, j] = C[j, i] = row[f"{p}{i}{p}{j}"]
    return C


def corr_vs_r(C):
    L = len(C)
    return np.array([np.mean([C[i, (i + r) % L] for i in range(L)]) for r in range(L)])


def params_from_name(path):
    m = dict(re.findall(r"(Jx|Jy|Jz|gamma|L)(-?[\d.]+)", path.stem))
    bc = "PBC" if "_PBC_" in path.stem else "OBC"
    return {k: float(v) for k, v in m.items()} | {"bc": bc}


def exact_file(prm):
    L = int(prm["L"])
    return EXACT_DIR / (
        f"correlators-exact-L{L}-{prm['bc']}-Jx{prm['Jx']}-Jy{prm['Jy']}-Jz{prm['Jz']}"
        f"-gamma{prm['gamma']}-dt0.05-init-gs.csv"
    )


def compare(tjm_path, fig_dir):
    prm = params_from_name(tjm_path)
    L = int(prm["L"])
    ex_path = exact_file(prm)
    if not ex_path.exists():
        print(f"no exact data for {tjm_path.name} (looked for {ex_path.name})")
        return
    tjm, ex = read_row(tjm_path), read_row(ex_path)

    for p in "XZ":
        if f"{p}0{p}1" not in tjm:
            continue
        Ct, Ce = corr_matrix(tjm, p, L), corr_matrix(ex, p, L)
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        for a, C, title in [(ax[0], Ct, "tensor jump"), (ax[1], Ce, "exact")]:
            im = a.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
            a.set(title=rf"{title}: $\langle {p}_i {p}_j\rangle$", xlabel="j", ylabel="i")
        fig.colorbar(im, ax=ax[:2], shrink=0.85)
        r = np.arange(L)
        ax[2].plot(r, corr_vs_r(Ct), "o-", label="tensor jump")
        ax[2].plot(r, corr_vs_r(Ce), "s--", label="exact")
        ax[2].axhline(0, color="gray", lw=0.5)
        ax[2].set(xlabel="r", ylabel=rf"$\langle {p}_i {p}_{{i+r}}\rangle$", title=f"max |diff| = {np.abs(Ct - Ce).max():.3f}")
        ax[2].legend()
        fig.suptitle(
            rf"L={L} {prm['bc']}, $J_x$={prm['Jx']:g}, $J_y$={prm['Jy']:g}, $J_z$={prm['Jz']:g}, $\gamma$={prm['gamma']:g}"
        )
        out = fig_dir / f"compare_{tjm_path.stem.removeprefix('tjm_')}_{p}{p}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved {out}")

    # one-site expectations, translation-averaged
    for p in "XYZ":
        if f"{p}0" in tjm:
            ex_val = np.mean([ex[f"{p}{i}{p}{i}"] for i in range(L)]) if f"{p}0{p}0" in ex else np.nan
            print(f"  <{p}>  tjm {np.mean([tjm[f'{p}{i}'] for i in range(L)]):+.4f}   exact {ex_val:+.4f}")


def main():
    fig_dir = HERE / "fig"
    fig_dir.mkdir(exist_ok=True)
    files = [Path(f) for f in sys.argv[1:]] or sorted((HERE / "data").glob("tjm_*.csv"))
    for f in files:
        compare(f.resolve(), fig_dir)


if __name__ == "__main__":
    main()
