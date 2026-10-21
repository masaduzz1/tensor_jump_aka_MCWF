"""Bond dimension chi(t) along TJM trajectories (same model/settings as xyz_ad_steady_state.py).

YAQS records per-trajectory MPS diagnostics (max bond, total bond, cost) at every time step when
sample_timesteps=True, but only exposes the trajectory average. We wrap its aggregation function to keep
the per-trajectory array as well (YAQS itself is unchanged). Only <Z_0> is measured, so the run is cheap.

Saves data/chi_trace_<stem>.npz and fig/chi_trace_<stem>.{pdf,png}: mean, 10-90% band, and max over
trajectories of the largest bond dimension vs t, plus the steady-state histogram.

Usage:
    conda activate tensorjump
    python chi_trace.py --Jx 3 --Jy 2 --num-traj 60 --max-bond-dim 128
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mqt.yaqs.simulator as yaqs_sim
from mqt.yaqs import AnalogSimParams, Hamiltonian, NoiseModel, Observable, Simulator, State

HERE = Path(__file__).resolve().parent
_PER_TRAJ = []
_orig_aggregate = yaqs_sim._aggregate_diagnostics


def _keep_per_traj(per_traj):
    _PER_TRAJ.append(np.array(per_traj, copy=True))  # shape (3, num_traj, n_times)
    return _orig_aggregate(per_traj)


yaqs_sim._aggregate_diagnostics = _keep_per_traj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=12)
    p.add_argument("--Jx", type=float, default=3.0)
    p.add_argument("--Jy", type=float, default=2.0)
    p.add_argument("--Jz", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--T", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--num-traj", type=int, default=60)
    p.add_argument("--preset", default="balanced")
    p.add_argument("--max-bond-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    L = a.L

    # same conventions as xyz_ad_steady_state.py (exact-data convention, |0...0> start)
    ham = Hamiltonian.heisenberg(L, -a.Jx / 2, -a.Jy / 2, -a.Jz / 2, bc="periodic")
    noise = NoiseModel([{"name": "raising", "sites": [i], "strength": a.gamma} for i in range(L)])
    params = AnalogSimParams(
        observables=[Observable("z", 0)], elapsed_time=a.T, dt=a.dt, num_traj=a.num_traj,
        preset=a.preset, max_bond_dim=a.max_bond_dim, order=1, sample_timesteps=True, random_seed=a.seed,
    )
    res = Simulator(parallel=True, max_workers=a.workers, show_progress=True).run(
        State(L, initial="zeros"), ham, params, noise
    )
    t = np.asarray(params.times)
    per = _PER_TRAJ[-1]
    chi, tot = per[1], per[2]  # (num_traj, n_times): max bond, total bond
    assert np.allclose(chi.mean(0), res.max_bond)

    stem = (f"Jx{a.Jx:g}_Jy{a.Jy:g}_Jz{a.Jz:g}_gamma{a.gamma:g}_L{L}_PBC_T{a.T:g}_dt{a.dt:g}"
            f"_ntraj{a.num_traj}_init-zeros_{a.preset}_order1_seed{a.seed}_conv-exact_chi{a.max_bond_dim}")
    (HERE / "data").mkdir(exist_ok=True)
    np.savez(HERE / "data" / f"chi_trace_{stem}.npz", times=t, max_bond=chi, total_bond=tot)

    k_pk = np.argmax(chi.mean(0))
    t_ss = min(8.0, 0.8 * a.T)  # "steady-state" window; shorter for short test runs
    ss = chi[:, t >= t_ss].ravel()
    print(f"mean max-bond: peak {chi.mean(0)[k_pk]:.1f} at t={t[k_pk]:.2f}; at t={t[-1]:g}: {chi.mean(0)[-1]:.1f}")
    print(f"largest bond in any trajectory at any time: {chi.max():.0f} (at t={t[np.argmax(chi.max(0))]:.2f})")
    print(f"steady state (t>={t_ss:g}, all trajectories): median {np.median(ss):.0f}, 90% {np.percentile(ss, 90):.0f}, "
          f"max {ss.max():.0f}; fraction of samples above 20/24/32: "
          f"{(ss > 20).mean():.2f}/{(ss > 24).mean():.2f}/{(ss > 32).mean():.2f}")
    print(f"fraction of trajectories exceeding 32 at some time: {(chi.max(1) > 32).mean():.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True, gridspec_kw={"width_ratios": [2.2, 1]})
    lo, hi = np.percentile(chi, [10, 90], axis=0)
    ax[0].fill_between(t, lo, hi, color="C0", alpha=0.25, lw=0, label="10–90% of trajectories")
    ax[0].plot(t, chi.mean(0), "C0", lw=1.6, label="mean over trajectories")
    ax[0].plot(t, chi.max(0), "C3", lw=1, label="max over trajectories")
    if 2 ** (L // 2) <= 2 * a.max_bond_dim:
        ax[0].axhline(2 ** (L // 2), color="gray", ls=":", lw=1, label=r"largest possible ($2^{L/2}$)")
    ax[0].axhline(a.max_bond_dim, color="k", ls="--", lw=0.8, label="cap")
    ax[0].set(xlabel="$t$", ylabel=r"largest bond dimension $\chi$", xlim=(0, a.T))
    ax[0].legend(fontsize=8)
    ax[1].hist(ss, bins=np.arange(0.5, ss.max() + 1.5), color="C0", alpha=0.8)
    ax[1].set(xlabel=r"$\chi$ (steady state, $t\geq8$)", ylabel="samples")
    for x, lab in zip(ax, "ab"):
        x.text(0.02, 0.96, f"({lab})", transform=x.transAxes, va="top", fontweight="bold")
    fig.suptitle(rf"$L$={L} PBC, $J_x$={a.Jx:g}, $J_y$={a.Jy:g}, $N$={a.num_traj}, cap {a.max_bond_dim}, SVD threshold {params.svd_threshold:g}")
    for ext in ("pdf", "png"):
        fig.savefig(HERE / "fig" / f"chi_trace_{stem}.{ext}", dpi=150)
    print(f"saved fig/chi_trace_{stem}.{{pdf,png}}")


if __name__ == "__main__":
    main()
