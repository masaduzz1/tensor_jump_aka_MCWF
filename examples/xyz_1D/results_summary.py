"""Summary figure + table of the conv-exact TJM runs (different trajectory counts) vs exact data.

Saves fig/results_summary_Jx3_Jy2.pdf (and .png): <X_0X_r>, <Z_0Z_r> vs r and S^XX(q), S^ZZ(q),
TJM with error bars and the exact steady state. Prints a LaTeX table body.

Usage:
    conda activate tensorjump
    python results_summary.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_exact_tjm import HERE, corr_matrix, corr_vs_r, exact_file, params_from_name, read_row
from structure_factor import structure_factor_pbc

STEM = "Jx3_Jy2_Jz1_gamma1_L12_PBC_T10_dt0.05_ntraj{n}_init-zeros_balanced_order1_seed{s}_conv-exact"
RUNS = [(300, 1234), (600, 5678), (2400, 9012)]
L = 12


def main():
    rows = []
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.4), constrained_layout=True)
    r = np.arange(L)
    ex = None
    for k, (n, s) in enumerate(RUNS):
        f = HERE / "data" / f"tjm_{STEM.format(n=n, s=s)}.csv"
        t, e = read_row(f), read_row(str(f).replace("tjm_", "tjm-err_"))
        ex = ex or read_row(exact_file(params_from_name(f)))
        mk = "osv"[k]  # open circle, square, triangle
        row = {"n": n, "seed": s}
        for col, p in enumerate("XZ"):
            Ct, Ce = corr_matrix(t, p, L), corr_matrix(ex, p, L)
            ct, ce = corr_vs_r(Ct), corr_vs_r(Ce)
            sem = np.array([0.0] + [e[f"{p}0{p}{j}"] for j in range(1, L)])
            ax[0, col].errorbar(r, ct, yerr=sem, fmt=mk, ms=6, mfc="none", mew=1.2, capsize=2, label=f"TJM, N={n}")
            q, S = structure_factor_pbc(Ct)
            ax[1, col].plot(q / np.pi, S, mk + "-", ms=6, mfc="none", mew=1.2, lw=1, label=f"TJM, N={n}")
            row[f"{p}dev"] = np.abs(ct[1:] - ce[1:]).max()
            row[f"{p}sem"] = sem[1:].mean()
            if p == "X":
                c, cex = ct[1:7], ce[1:7]
                row["ratio"] = c @ cex / (cex @ cex)
                row["dratio"] = (np.abs(cex) @ sem[1:7]) / (cex @ cex)
                row["Speak"] = S.max()
            else:
                row["ZZmean"] = ct[1:].mean()
        row["Z"] = np.mean([t[f"Z{i}"] for i in range(L)])
        row["dZ"] = e["Z0"]
        rows.append(row)

    for col, p in enumerate("XZ"):
        Ce = corr_matrix(ex, p, L)
        r0 = 1 if p == "Z" else 0  # ZZ panel is zoomed; skip r = 0 (= 1)
        ax[0, col].plot(r[r0:], corr_vs_r(Ce)[r0:], "k--", lw=1.3, label="exact", zorder=5)
        q, S = structure_factor_pbc(Ce)
        ax[1, col].plot(q / np.pi, S, "k--", lw=1.3, label="exact", zorder=5)
        ax[0, col].axhline(0, color="gray", lw=0.5)
        ax[0, col].set(xlabel="$r$", ylabel=rf"$\langle {p}_i {p}_{{i+r}}\rangle$")
        ax[1, col].set(xlabel=r"$q/\pi$", ylabel=rf"$S^{{{p}{p}}}(q)$", xlim=(-1, 1))
        if p == "Z":  # zoom: r = 0 is 1 and hides the detail
            ax[0, col].set_ylim(0.6, 0.7)
            ax[0, col].set_xlim(0.5, L - 0.5)
        for a in ax[:, col]:
            a.legend(fontsize=8)
    for a, lab in zip(ax.flat, "abcd"):
        a.text(0.02, 0.96, f"({lab})", transform=a.transAxes, va="top", fontweight="bold")
    fig.suptitle(r"L=12 PBC, $J_x$=3, $J_y$=2, $J_z$=1, $\gamma$=1, steady state (t=10)")
    for ext in ("pdf", "png"):
        fig.savefig(HERE / "fig" / f"results_summary_Jx3_Jy2.{ext}", dpi=150)
    print("saved fig/results_summary_Jx3_Jy2.{pdf,png}")

    ex_Z = np.mean([ex[f"Z{i}Z{i}"] for i in range(L)])
    ex_ZZ = corr_vs_r(corr_matrix(ex, "Z", L))[1:].mean()
    ex_S = structure_factor_pbc(corr_matrix(ex, "X", L))[1].max()
    for w in rows:
        print(f"{w['n']} & {w['seed']} & {w['ratio']:.3f}\\,$\\pm$\\,{w['dratio']:.3f} & {w['Xdev']:.3f} ({w['Xsem']:.3f}) & "
              f"{w['Speak']:.2f} & {w['ZZmean']:.3f} ({w['Zsem']:.3f}) & {w['Z']:.4f}\\,$\\pm$\\,{w['dZ']:.4f} \\\\")
    print(f"exact & -- & 1 & -- & {ex_S:.2f} & {ex_ZZ:.3f} & {ex_Z:.4f} \\\\")


if __name__ == "__main__":
    main()
