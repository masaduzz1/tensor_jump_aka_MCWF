"""Cross-check the damping in YAQS: TJM vs MCWF vs Lindblad vs an independent solver, L=6.

Same problem for all four:
    H = sum_i (Jx X_i X_{i+1} + Jy Y_i Y_{i+1} + Jz Z_i Z_{i+1}),  PBC
    L_i = sqrt(gamma) |0><1|_i   (YAQS "lowering"), initial state = ground state of H
Reference: Lindblad equation solved here with dense numpy/scipy (vectorized rho,
exact propagator exp(L dt)), independent of YAQS.

Outputs data/check_L6_*.npz and fig/check_L6_*.png (time traces of <Z_0>, <Z_0Z_1>, <X_0X_1>).

Usage:
    conda activate tensorjump
    python check_L6_solvers.py --num-traj 500
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
import scipy.linalg
from mqt.yaqs import AnalogSimParams, Hamiltonian, NoiseModel, Observable, Simulator, State

from xyz_ad_steady_state import PAULI, vector_to_mps_tensors

HERE = Path(__file__).resolve().parent
OBS = {"Z0": ("z", 0), "Z0Z1": ("ZZ", [0, 1]), "X0X1": ("XX", [0, 1])}


def op_at(mats, L):
    """Tensor product with site 0 least significant (YAQS / Qiskit order)."""
    out = np.array([[1.0 + 0j]])
    for site in reversed(range(L)):
        out = np.kron(out, mats.get(site, np.eye(2)))
    return out


def reference_lindblad(L, Jx, Jy, Jz, gamma, psi0, times):
    """Independent dense Lindblad solver. Returns {label: trace} and H."""
    X, Y, Z = PAULI["X"], PAULI["Y"], PAULI["Z"]
    H = sum(
        J * op_at({i: P, (i + 1) % L: P}, L)
        for i in range(L)
        for J, P in [(Jx, X), (Jy, Y), (Jz, Z)]
    )
    sm = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1|
    Ls = [np.sqrt(gamma) * op_at({i: sm}, L) for i in range(L)]
    d = 2**L
    I = np.eye(d)
    # column-stacking vec: vec(A rho B) = (B^T kron A) vec(rho)
    gen = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for Lk in Ls:
        LdL = Lk.conj().T @ Lk
        gen += np.kron(Lk.conj(), Lk) - 0.5 * np.kron(I, LdL) - 0.5 * np.kron(LdL.T, I)
    prop = scipy.linalg.expm(gen * (times[1] - times[0]))
    obs = {
        "Z0": op_at({0: Z}, L),
        "Z0Z1": op_at({0: Z, 1: Z}, L),
        "X0X1": op_at({0: X, 1: X}, L),
    }
    rho = np.outer(psi0, psi0.conj()).reshape(-1, order="F")
    out = {k: [] for k in obs}
    for _ in times:
        R = rho.reshape(d, d, order="F")
        for k, O in obs.items():
            out[k].append(np.real(np.trace(O @ R)))
        rho = prop @ rho
    return {k: np.array(v) for k, v in out.items()}, H


def yaqs_observables():
    obs = []
    for name, (op, sites) in OBS.items():
        if op in ("XX", "ZZ"):
            P = PAULI[op[0]]
            obs.append(Observable(np.kron(P, P), sites))
        else:
            obs.append(Observable(op, sites))
    return obs


def run_yaqs(state, ham, noise, T, dt, num_traj, workers, **kw):
    params = AnalogSimParams(
        observables=yaqs_observables(), elapsed_time=T, dt=dt, num_traj=num_traj,
        sample_timesteps=True, random_seed=1234, **kw,
    )
    res = Simulator(parallel=True, max_workers=workers, show_progress=False).run(state, ham, params, noise)
    out = {}
    for k, name in enumerate(OBS):
        tr = np.real(np.asarray(res.trajectories[k])).reshape(-1, len(params.times))
        n = tr.shape[0]
        out[name] = (tr.mean(0), tr.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(tr.shape[1]))
    return np.asarray(params.times), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=6)
    p.add_argument("--Jx", type=float, default=3.0)
    p.add_argument("--Jy", type=float, default=2.0)
    p.add_argument("--Jz", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--T", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--num-traj", type=int, default=500)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    L = a.L

    ham = Hamiltonian.heisenberg(L, -a.Jx, -a.Jy, -a.Jz, bc="periodic")
    noise = NoiseModel([{"name": "lowering", "sites": [i], "strength": a.gamma} for i in range(L)])

    # ground state from YAQS's H; checked against the independently built H below
    w, V = np.linalg.eigh(ham.to_matrix())
    psi0 = V[:, 0]
    times = np.round(np.arange(0, a.T + a.dt / 2, a.dt), 10)

    ref, H_ref = reference_lindblad(L, a.Jx, a.Jy, a.Jz, a.gamma, psi0, times)
    print(f"max |H_yaqs - H_ref| = {np.abs(ham.to_matrix() - H_ref).max():.2e}, E0 gap = {w[1] - w[0]:.4f}")

    results = {"reference (numpy Lindblad)": {k: (v, np.zeros_like(v)) for k, v in ref.items()}}
    solvers = [
        ("YAQS Lindblad", State(density_matrix=np.outer(psi0, psi0.conj())), 1),
        ("YAQS MCWF", State(vector=psi0), a.num_traj),
        ("YAQS TJM", State(tensors=vector_to_mps_tensors(psi0, L)), a.num_traj),
    ]
    for name, state, ntraj in solvers:
        t, out = run_yaqs(state, ham, noise, a.T, a.dt, ntraj, a.workers, preset="accurate")
        assert np.allclose(t, times)
        results[name] = out
        print(f"done: {name}")

    stem = f"L{L}_Jx{a.Jx:g}_Jy{a.Jy:g}_Jz{a.Jz:g}_gamma{a.gamma:g}_T{a.T:g}_dt{a.dt:g}_ntraj{a.num_traj}"
    (HERE / "data").mkdir(exist_ok=True)
    np.savez(HERE / "data" / f"check_{stem}.npz", times=times,
             **{f"{s}|{o}|{m}": results[s][o][j] for s in results for o in OBS for j, m in enumerate(["mean", "sem"])})

    print("\n t     " + "".join(f"{s[:16]:>18}" for s in results))
    for o in OBS:
        print(o)
        for tk in [0.05, 0.1, 0.2, 0.5, 1.0, 3.0, a.T]:
            k = int(round(tk / a.dt))
            row = "".join(
                f"{results[s][o][0][k]:>10.4f}" + (f"({results[s][o][1][k]:.3f})" if results[s][o][1][k] else " " * 7)
                + " "
                for s in results
            )
            print(f" {tk:<5g}{row}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, o in zip(axes, OBS):
        for k, (s, style) in enumerate(zip(results, ["k-", "C1--", "C2-", "C0-"])):
            m, e = results[s][o]
            ax.plot(times, m, style, lw=1.2, label=s)
            if e.any():
                ax.fill_between(times, m - e, m + e, color=style[:2], alpha=0.2, lw=0)
        ax.set(xlabel="t", title=f"<{o}>")
        ax.legend(fontsize=8)
    fig.suptitle(rf"L={L} PBC, $J_x$={a.Jx:g}, $J_y$={a.Jy:g}, $J_z$={a.Jz:g}, $\gamma$={a.gamma:g}, dt={a.dt:g}, ntraj={a.num_traj}")
    (HERE / "fig").mkdir(exist_ok=True)
    out = HERE / "fig" / f"check_{stem}.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
