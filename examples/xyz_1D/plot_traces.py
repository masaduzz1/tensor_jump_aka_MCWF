"""Plot TJM time traces (data/traces_*.npz, from --traces runs) against the exact time series.

Saves fig/traces_<stem>.png: <X_0 X_r>, <Z_0 Z_r> for r = 1..L/2 and <Z_0>, with the
TJM standard error as a shaded band and the exact data as dashed lines.

Usage:
    conda activate tensorjump
    python plot_traces.py              # all traces files in data/
    python plot_traces.py data/traces_....npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_exact_tjm import HERE, exact_file, params_from_name


def plot(path, fig_dir):
    prm = params_from_name(path)
    L = int(prm["L"])
    z = np.load(path)
    labels = list(z["labels"])
    t, mean, sem = z["times"], z["mean"], z["sem"]

    ex_path = exact_file(prm)
    ex = None
    if ex_path.exists():
        header = open(ex_path).readline().strip().split(",")
        ex = dict(zip(header, np.loadtxt(ex_path, delimiter=",", skiprows=1).T))

    panels = [(p, [f"{p}0{p}{r}" for r in range(1, L // 2 + 1)]) for p in "XZ"]
    panels.append(("Z", ["Z0"]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, (p, labs) in zip(axes, panels):
        for k, lab in enumerate(labs):
            if lab not in labels:
                continue
            c = f"C{k}"
            i = labels.index(lab)
            ax.plot(t, mean[i], color=c, label=lab)
            ax.fill_between(t, mean[i] - sem[i], mean[i] + sem[i], color=c, alpha=0.25, lw=0)
            # exact files store <Z_i> in the Z_iZ_i column
            ex_lab = lab if len(lab) > 2 else f"{lab}{lab}"
            if ex is not None and ex_lab in ex:
                ax.plot(ex["t"], ex[ex_lab], "--", color=c, lw=1)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set(xlabel="t", title=f"{'<Z_0>' if labs == ['Z0'] else f'<{p}_0 {p}_r>'}  (solid: TJM, dashed: exact)")
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(
        rf"L={L} {prm['bc']}, $J_x$={prm['Jx']:g}, $J_y$={prm['Jy']:g}, $J_z$={prm['Jz']:g}, $\gamma$={prm['gamma']:g}"
    )
    out = fig_dir / f"traces_{path.stem.removeprefix('traces_')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main():
    fig_dir = HERE / "fig"
    fig_dir.mkdir(exist_ok=True)
    files = [Path(f) for f in sys.argv[1:]] or sorted((HERE / "data").glob("traces_*.npz"))
    for f in files:
        plot(f.resolve(), fig_dir)


if __name__ == "__main__":
    main()
