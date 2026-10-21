"""Steady-state <X_i X_j> of the dissipative XYZ chain via YAQS tensor jump method.

Model (matches the exact data in data-dd-heisenberg/data-exact; fitted on the L=10
exact files with check_exact_normalization.py):
    H   = (1/2) sum_i (Jx X_i X_{i+1} + Jy Y_i Y_{i+1} + Jz Z_i Z_{i+1})   (Pauli matrices)
    L_i = sqrt(gamma) |1><0|_i   (damping toward <Z> = -1, QuTiP sigmam() convention)

Only the final time is recorded (sample_timesteps=False). From the L=12 exact
data in data-dd-heisenberg/data-exact (gamma=1), XX correlators are converged
to 1e-2 by t~4 for most (Jx, Jy) and by t~8.2 in the slowest cases, so
T=10/gamma is the default.

Usage:
    conda activate tensorjump
    python xyz_ad_steady_state.py --Jx 1.0 --Jy 0.0 --num-traj 300
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Pin numba to one thread before YAQS imports it; otherwise YAQS worker processes
# can fail with "Cannot set NUMBA_NUM_THREADS to a different value ...".
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
from mqt.yaqs import AnalogSimParams, Hamiltonian, NoiseModel, Observable, Simulator, State

X = np.array([[0, 1], [1, 0]], dtype=complex)

PAULI = {
    "X": X,
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pair_observable(p, i, j):
    # Non-neighbouring pairs are handled natively by MPS.local_expect on this branch (operator Schmidt decomposition).
    return Observable(np.kron(PAULI[p], PAULI[p]), [i, j])


def file_stem(L, Jx, Jy, Jz, gamma, bc, T, dt, num_traj, init="gs", preset="accurate", order=1, seed=1234):
    bc = "PBC" if bc == "periodic" else "OBC"
    return (
        f"Jx{Jx:g}_Jy{Jy:g}_Jz{Jz:g}_gamma{gamma:g}_L{L}_{bc}_T{T:g}_dt{dt:g}"
        f"_ntraj{num_traj}_init-{init}_{preset}_order{order}_seed{seed}_conv-exact"
    )


def vector_to_mps_tensors(psi, L, d=2):
    """Exact (untruncated) MPS of a dense state vector, in YAQS conventions.

    YAQS vectors are site-0-least-significant and MPS tensors have shape
    (physical, left bond, right bond).
    """
    # C-order reshape puts site L-1 first; reverse so axis k is site k.
    rest = psi.reshape([d] * L).transpose(list(range(L))[::-1]).reshape(1, -1)
    tensors = []
    for _ in range(L - 1):
        chi = rest.shape[0]
        u, s, vh = np.linalg.svd(rest.reshape(chi * d, -1), full_matrices=False)
        keep = s > 1e-14 * s[0]
        u, s, vh = u[:, keep], s[keep], vh[keep]
        tensors.append(u.reshape(chi, d, -1).transpose(1, 0, 2))
        rest = s[:, None] * vh
    tensors.append(rest.reshape(rest.shape[0], d, 1).transpose(1, 0, 2))
    return tensors


def ground_state(ham, L):
    """Ground state of the (closed) Hamiltonian by sparse exact diagonalization.

    Matches the "init-gs" initial state of the exact data. L=12 means a 4096-dim
    vector, so this is cheap.
    """
    from scipy.sparse.linalg import eigsh

    evals, evecs = eigsh(ham.to_sparse_matrix(), k=2, which="SA")
    order = np.argsort(evals)
    if abs(evals[order[1]] - evals[order[0]]) < 1e-8:
        print(f"warning: ground state is (near-)degenerate, E0={evals[order[0]]:.8f}, E1={evals[order[1]]:.8f}")
    psi = evecs[:, order[0]]
    return State(tensors=vector_to_mps_tensors(psi, L))


def run(L, Jx, Jy, Jz, gamma, bc, T, dt, num_traj, preset, seed, singles="XYZ", pairs="XZ",
        init="gs", order=1, show_progress=True, traces=False, workers=6,
        max_bond_dim=None, svd_threshold=None):
    """Run TJM and return (labels, times, mean, sem).

    mean[k, t] / sem[k, t]: trajectory average and its standard error
    (std over trajectories / sqrt(num_traj)) of observable labels[k] at times[t].
    With traces=False only the final time is kept (t axis has length 1).
    With traces=True every dt is recorded, and to keep that cheap only the pairs
    (0, r) are measured: under PBC <P_i P_{i+r}> does not depend on i.
    """
    # Closed-system part. YAQS builds H = -sum_i (Jx XX + Jy YY + Jz ZZ) with Pauli
    # matrices. The exact data uses H = +(1/2) sum_i (Jx XX + Jy YY + Jz ZZ), so pass
    # -J/2. bc="periodic" adds the (L-1, 0) bond.
    ham = Hamiltonian.heisenberg(L, -Jx / 2, -Jy / 2, -Jz / 2, bc=bc)

    # Dissipation: one jump operator sqrt(gamma) |1><0| per site. YAQS "raising" is
    # |1><0| (|0> -> |1>), which with Z = diag(1, -1) drives <Z> -> -1, as in the exact
    # data (QuTiP's sigmam()). "lowering" (|0><1|) would give the same XX, YY, ZZ
    # correlators and only flip the sign of <Z>. "strength" is the Lindblad rate gamma
    # itself; YAQS takes the square root internally.
    noise = NoiseModel([{"name": "raising", "sites": [i], "strength": gamma} for i in range(L)])

    # Initial state. "gs": ground state of H (as in the exact data, init-gs);
    # "zeros": the product state |0...0>, the vacuum of the damping.
    state = ground_state(ham, L) if init == "gs" else State(L, initial="zeros")

    labels, obs = [], []
    for p in singles:
        for i in range(L):
            labels.append(f"{p}{i}")
            obs.append(Observable(p.lower(), i))
    for p in pairs:
        for i in range(1 if traces else L):
            for j in range(i + 1, L):
                labels.append(f"{p}{i}{p}{j}")
                obs.append(pair_observable(p, i, j))

    params = AnalogSimParams(
        # Measured at the end of every trajectory, then averaged over trajectories.
        observables=obs,
        # Total evolution time; must be an integer multiple of dt.
        elapsed_time=T,
        # Time step. Each step does: (1) evolve with the non-Hermitian effective
        # Hamiltonian H_eff = H - (i/2) sum_k L_k^dag L_k using TDVP on the MPS,
        # (2) renormalize and apply a random quantum jump L_k with probability set
        # by the norm loss in that step. The splitting of H and the dissipator
        # gives an O(dt) error per unit time (order=1) or O(dt^2) (order=2).
        dt=dt,
        # Number of independent stochastic trajectories. Statistical error of an
        # expectation value scales as 1/sqrt(num_traj). Overrides the preset value.
        num_traj=num_traj,
        # Preset numerical tolerances (explicit arguments override it):
        #   fast:     svd_threshold 1e-3,  max_bond_dim 16,   krylov_tol 1e-3
        #   balanced: svd_threshold 1e-6,  max_bond_dim 128,  krylov_tol 1e-4
        #   accurate: svd_threshold 1e-9,  max_bond_dim 4096, krylov_tol 1e-6
        #   exact:    svd_threshold 1e-13, no bond cap,       krylov_tol 1e-12
        # svd_threshold: discarded weight allowed per SVD truncation (MPS compression).
        # max_bond_dim: hard cap on MPS bond dimension (for L=12 the largest possible
        #   bond is 2^6 = 64, so 128 or more is effectively no cap).
        # krylov_tol: accuracy of the Lanczos matrix exponential inside TDVP.
        # The preset does not remove time-step (dt) or sampling (num_traj) error.
        preset=preset,
        # Explicit overrides of the preset (only passed when given on the command line).
        **({"max_bond_dim": max_bond_dim} if max_bond_dim is not None else {}),
        **({"svd_threshold": svd_threshold} if svd_threshold is not None else {}),
        # TJM integrator order: 1 = first-order (Lie) splitting of unitary and
        # dissipative parts; 2 = second-order (Strang) splitting, smaller dt error.
        order=order,
        # How SVD truncation is decided; "discarded_weight" drops the smallest
        # singular values as long as their summed squares stay below svd_threshold.
        trunc_mode="discarded_weight",
        # MPS time-evolution algorithm. "tdvp" = time-dependent variational principle
        # (default); "bug" = basis-update-and-Galerkin integrator.
        evolution_mode="tdvp",
        # TDVP variant: "2site" can grow the bond dimension (needed when starting
        # from a low-entanglement state); "1site" keeps it fixed; "dynamic" switches.
        tdvp_mode="2site",
        # Each dt is split into this many symmetric TDVP sweeps.
        tdvp_sweeps=1,
        # True: measure after every dt (time traces, to check convergence to the
        # steady state). False: only the final time is measured.
        sample_timesteps=traces,
        # Seed for the jump random numbers; makes runs reproducible. Change it to get
        # an independent sample (useful to estimate the statistical error).
        random_seed=seed,
    )
    # Trajectories are independent, so they run in a process pool: parallel=True,
    # max_workers = number of worker processes (each worker runs whole trajectories,
    # with BLAS/numba limited to one thread to avoid oversubscription).
    sim = Simulator(parallel=True, max_workers=workers, show_progress=show_progress)
    res = sim.run(state, ham, params, noise)

    # res.trajectories[k]: per-trajectory values, shape (num_traj, n_times).
    tr = np.real(np.array([np.atleast_2d(np.asarray(t).reshape(num_traj, -1)) for t in res.trajectories]))
    mean = tr.mean(axis=1)
    sem = tr.std(axis=1, ddof=1) / np.sqrt(num_traj)
    times = np.asarray(params.times) if traces else np.array([T])
    # YAQS MPS diagnostics, averaged over trajectories at each recorded time:
    # max_bond = largest bond dimension in the MPS, total_bond = sum over all bonds,
    # runtime_cost = YAQS's contraction-cost estimate.
    diag = {
        "max_bond": np.atleast_1d(np.asarray(res.max_bond, dtype=float)),
        "total_bond": np.atleast_1d(np.asarray(res.total_bond, dtype=float)),
        "runtime_cost": np.atleast_1d(np.asarray(res.runtime_cost, dtype=float)),
        "cap": params.max_bond_dim,
        "svd_threshold": params.svd_threshold,
    }
    return labels, times, mean, sem, diag


def fill_pairs_by_translation(row, L, pairs):
    """Fill all P_iP_j (i<j) from P_0P_r using PBC translation invariance."""
    for p in pairs:
        for i in range(L):
            for j in range(i + 1, L):
                row.setdefault(f"{p}{i}{p}{j}", row[f"{p}0{p}{j - i}"])
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=12)
    p.add_argument("--Jx", type=float, default=1.0)
    p.add_argument("--Jy", type=float, default=0.0)
    p.add_argument("--Jz", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--bc", default="periodic", choices=["periodic", "open"])
    p.add_argument("--T", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--num-traj", type=int, default=300)
    p.add_argument("--preset", default="accurate", choices=["fast", "balanced", "accurate", "exact"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--init", default="gs", choices=["gs", "zeros"], help="initial state")
    p.add_argument("--order", type=int, default=1, choices=[1, 2], help="TJM splitting order")
    p.add_argument("--singles", default="XYZ", help="one-site Paulis to record")
    p.add_argument("--pairs", default="XZ", help="two-site PP correlators to record")
    p.add_argument("--traces", action="store_true", help="record time traces (pairs (0,r) only)")
    p.add_argument("--workers", type=int, default=6, help="parallel worker processes")
    p.add_argument("--max-bond-dim", type=int, default=None, help="override the preset bond-dimension cap")
    p.add_argument("--svd-threshold", type=float, default=None, help="override the preset SVD threshold")
    p.add_argument("--out-dir", default="data")
    a = p.parse_args()

    labels, times, mean, sem, diag = run(
        a.L, a.Jx, a.Jy, a.Jz, a.gamma, a.bc, a.T, a.dt, a.num_traj, a.preset, a.seed,
        a.singles, a.pairs, a.init, a.order, traces=a.traces, workers=a.workers,
        max_bond_dim=a.max_bond_dim, svd_threshold=a.svd_threshold,
    )

    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    stem = file_stem(a.L, a.Jx, a.Jy, a.Jz, a.gamma, a.bc, a.T, a.dt, a.num_traj, a.init, a.preset, a.order, a.seed)
    if a.max_bond_dim is not None:
        stem += f"_chi{a.max_bond_dim}"
    if a.svd_threshold is not None:
        stem += f"_svd{a.svd_threshold:g}"
    row = dict(zip(labels, mean[:, -1]))
    err = dict(zip(labels, sem[:, -1]))
    if a.traces:
        fill_pairs_by_translation(row, a.L, a.pairs)
        fill_pairs_by_translation(err, a.L, a.pairs)

    # tjm_*.csv: steady-state means, columns like X0, Y0, Z0, X0X1, Z0Z1
    # tjm-err_*.csv: their standard errors (same columns)
    for prefix, d in [("tjm", row), ("tjm-err", err)]:
        path = out / f"{prefix}_{stem}.csv"
        with open(path, "w") as f:
            f.write(",".join(d) + "\n")
            f.write(",".join(repr(float(v)) for v in d.values()) + "\n")
        print(f"saved {path}")
    path = out / f"diag_{stem}.npz"  # bond-dimension diagnostics
    np.savez(path, times=times, **diag)
    print(f"saved {path}")
    print(f"bond cap {diag['cap']}, svd_threshold {diag['svd_threshold']:g}; "
          f"mean max bond dim at t={times[-1]:g}: {diag['max_bond'][-1]:.1f}, mean total bond dim: {diag['total_bond'][-1]:.1f}")
    if a.traces:  # time traces of the measured observables
        path = out / f"traces_{stem}.npz"
        np.savez(path, labels=np.array(labels), times=times, mean=mean, sem=sem)
        print(f"saved {path}")

if __name__ == "__main__":
    main()
