# Dissipative XYZ chain with amplitude damping (1D, periodic)

Steady-state correlators of

    H   = (1/2) sum_i (Jx X_i X_{i+1} + Jy Y_i Y_{i+1} + Jz Z_i Z_{i+1})      (periodic)
    L_i = sqrt(gamma) |1><0|_i                                             (damping toward <Z> = -1)

computed with the YAQS tensor jump method (TJM), for L = 12 (compared with exact data) and L = 40.
In YAQS this is `Hamiltonian.heisenberg(L, -Jx/2, -Jy/2, -Jz/2, bc="periodic")` with `"raising"` noise.

Long-range two-site observables `<A_i B_j>` (|i - j| > 1) use `MPS.local_expect` on this branch, which
evaluates them by an operator Schmidt decomposition and full overlaps (see `src/mqt/yaqs/core/data_structures/mps.py`).

## Scripts

| script | purpose |
|---|---|
| `xyz_ad_steady_state.py` | main driver: runs TJM, writes `data/tjm_<stem>.csv`, `tjm-err_<stem>.csv` (standard errors), `diag_<stem>.npz` (bond dimensions); `--traces` also writes time traces |
| `compare_exact_tjm.py` | TJM vs exact `<X_iX_{i+r}>`, `<Z_iZ_{i+r}>` |
| `structure_factor.py` | `S^XX(q)` from the correlation matrix (PBC) |
| `compare_runs.py` | several runs (bond-dimension / trajectory scans) vs exact with error bands |
| `results_summary.py` | summary figure for (Jx, Jy) = (3, 2), N = 300/600/2400 |
| `chi_trace.py` | largest bond dimension chi(t) of every trajectory |
| `plot_tjm_only.py` | correlators and `S^XX(q)` for runs without exact data (L = 40) |
| `plot_dt_compare.py` | overlay of runs differing in one setting (dt, bond cap) |
| `plot_traces.py` | time traces vs exact |
| `check_L6_solvers.py` | L = 6: TJM / MCWF / Lindblad vs a dense Lindblad solver |
| `check_exact_normalization.py` | fixes the convention of the exact data (L = 10) |

Example:

    python xyz_ad_steady_state.py --L 12 --Jx 3 --Jy 2 --num-traj 300 --preset balanced --init zeros --order 1
    python xyz_ad_steady_state.py --L 40 --Jx 1 --Jy 0 --num-traj 128 --preset balanced --init zeros --max-bond-dim 16

Default settings used: dt = 0.05, T = 10, |0...0> start, Lie splitting (`order=1`), 6 workers.

## Data

`fig/` holds the figures of the study. The numerical data (`data/`) and the exact reference data are not
included; `compare_exact_tjm.py` and `check_exact_normalization.py` expect the exact files in
`../data-dd-heisenberg/data-dd-heisenberg/data-exact/` relative to this folder.

## Main findings

* L = 12: with the convention above, TJM agrees with exact data within error bars for XX at (3, 2) (N = 2400)
  and for XX, ZZ at (+-1, 0); at (3, 2) ZZ and |<Z>| are ~0.008 (2-3 sigma) high, likely an O(dt) bias.
* (1, 0) and (-1, 0) are equivalent for even L (U = prod_odd Z_i): XX(r) -> (-1)^r XX(r).
* Bond dimension: at L = 12 chi peaks at ~38 (t ~ 1.4) and relaxes to ~20 at (3, 2), and saturates at ~24
  at (+-1, 0). At L = 40 it keeps growing to t = 10 (mean 37 with a cap of 64).
* L = 40, (1, 0): a bond cap of 16 and 64 give the same correlators within 0.006 (same seed);
  dt = 0.1 vs 0.05 shifts <Z> by ~0.025 (~1 sigma). Runtime: 68 min (cap 16) vs 8.3 h (cap 64), 128 trajectories, 6 cores.
