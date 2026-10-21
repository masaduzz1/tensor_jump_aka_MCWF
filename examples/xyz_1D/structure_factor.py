"""Static structure factor S^PP(q) from tensor jump vs exact steady-state correlators.

S(q) = sum_r C(r) e^{-iqr},  C(r) = (1/L) sum_i <P_i P_{i+r}>   (PBC, diagonal = 1),
same convention as data-exact-plot/plot_correlator.py::structure_factor_pbc.

Saves fig/sq_<stem>.png with S^XX(q) (and S^ZZ(q) if present), TJM and exact on
the same axes; q = pi is repeated at -pi so the curve closes.

Usage:
    conda activate tensorjump
    python structure_factor.py            # all files in data/
    python structure_factor.py data/tjm_Jx3_Jy2_....csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_exact_tjm import HERE, corr_matrix, corr_vs_r, exact_file, params_from_name, read_row


def structure_factor_pbc(C):
    """Return (q, S(q)) with q in [-pi, pi], endpoint pi repeated at -pi."""
    L = len(C)
    S = np.real(np.fft.fftshift(np.fft.fft(corr_vs_r(C))))
    q = np.fft.fftshift(np.fft.fftfreq(L)) * 2 * np.pi
    if L % 2 == 0:  # q runs -pi .. pi-2pi/L; append +pi = -pi
        q, S = np.append(q, np.pi), np.append(S, S[0])
    return q, S


def plot(tjm_path, fig_dir):
    prm = params_from_name(tjm_path)
    L = int(prm["L"])
    ex_path = exact_file(prm)
    if not ex_path.exists():
        print(f"no exact data for {tjm_path.name}")
        return
    tjm, ex = read_row(tjm_path), read_row(ex_path)
    ps = [p for p in "XZ" if f"{p}0{p}1" in tjm]

    fig, axes = plt.subplots(1, len(ps), figsize=(5 * len(ps), 3.8), constrained_layout=True, squeeze=False)
    for ax, p in zip(axes[0], ps):
        for row, style, label in [(tjm, "o-", "tensor jump"), (ex, "s--", "exact")]:
            q, S = structure_factor_pbc(corr_matrix(row, p, L))
            ax.plot(q / np.pi, S, style, label=label)
        ax.set(xlabel=r"$q/\pi$", ylabel=rf"$S^{{{p}{p}}}(q)$")
        ax.set_xlim(-1, 1)
        ax.legend(loc="best", fontsize=9, framealpha=0.6)
    fig.suptitle(
        rf"L={L} {prm['bc']}, $J_x$={prm['Jx']:g}, $J_y$={prm['Jy']:g}, $J_z$={prm['Jz']:g}, $\gamma$={prm['gamma']:g}"
    )
    out = fig_dir / f"sq_{tjm_path.stem.removeprefix('tjm_')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main():
    fig_dir = HERE / "fig"
    fig_dir.mkdir(exist_ok=True)
    files = [Path(f) for f in sys.argv[1:]] or sorted((HERE / "data").glob("tjm_*.csv"))
    for f in files:
        plot(f.resolve(), fig_dir)


if __name__ == "__main__":
    main()
