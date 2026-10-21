"""Find the model convention behind the exact data using the L=10 exact files.

Solves the Lindblad equation (dense rho, RK4) for candidate conventions
    H = s * sum_i (Jx XX + Jy YY + Jz ZZ)   (PBC, Pauli matrices)
    jumps sqrt(g * gamma) * L_i,  L = |1><0| (drives <Z> -> -1, as in the exact data)
for s in {+1, -1, 1/2, 1/4} and g in {1/2, 1, 2}, starting from |+>^L, and reports
the max deviation from the exact file over 0 <= t <= t_fit.

Usage:
    conda activate tensorjump
    python check_exact_normalization.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
EXACT_DIR = HERE.parent / "data-dd-heisenberg" / "data-dd-heisenberg" / "data-exact"

X = sp.csr_matrix([[0, 1], [1, 0]], dtype=complex)
Y = sp.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
Z = sp.csr_matrix([[1, 0], [0, -1]], dtype=complex)
UP = sp.csr_matrix([[0, 0], [1, 0]], dtype=complex)  # |1><0|


def op_at(mats, L):
    """Site 0 least significant."""
    out = sp.identity(1, dtype=complex, format="csr")
    for site in reversed(range(L)):
        out = sp.kron(out, mats.get(site, sp.identity(2, dtype=complex)), format="csr")
    return out


def evolve(L, Jx, Jy, Jz, gamma, s, g, t_out, h=0.01):
    H = s * sum(J * op_at({i: P, (i + 1) % L: P}, L) for i in range(L) for J, P in [(Jx, X), (Jy, Y), (Jz, Z)])
    Ls = [np.sqrt(g * gamma) * op_at({i: UP}, L) for i in range(L)]
    Heff = (H - 0.5j * sum(Lk.conj().T @ Lk for Lk in Ls)).tocsr()

    def rhs(r):
        a = -1j * (Heff @ r)
        out = a + a.conj().T
        for Lk in Ls:
            out += Lk @ (Lk @ r.conj().T).conj().T  # L r L^dag = L (L r^dag)^dag
        return out

    plus = np.ones(2**L) / np.sqrt(2**L)
    rho = np.outer(plus, plus).astype(complex)
    obs = {"Z0Z0": op_at({0: Z}, L), "X0X1": op_at({0: X, 1: X}, L), "Z0Z1": op_at({0: Z, 1: Z}, L),
           "X0X2": op_at({0: X, 2: X}, L)}
    res = {k: [] for k in obs}
    t, k_out = 0.0, 0
    n_sub = int(round((t_out[1] - t_out[0]) / h))
    for k_out in range(len(t_out)):
        for k, O in obs.items():
            res[k].append(np.real((O @ rho).trace()))
        if k_out == len(t_out) - 1:
            break
        for _ in range(n_sub):
            k1 = rhs(rho)
            k2 = rhs(rho + 0.5 * h * k1)
            k3 = rhs(rho + 0.5 * h * k2)
            k4 = rhs(rho + h * k3)
            rho = rho + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return {k: np.array(v) for k, v in res.items()}


def load(name):
    f = EXACT_DIR / name
    hdr = open(f).readline().strip().split(",")
    d = np.loadtxt(f, delimiter=",", skiprows=1)
    return {c: d[:, i] for i, c in enumerate(hdr)}


def main():
    L, t_fit = 10, 1.0
    cases = [
        ("correlators-exact-L10-PBC-Jx4.0-Jy2.0-Jz1.0-gamma1.0-dt0.05-init-all-plus.csv", 4.0, 2.0),
        ("correlators-exact-L10-PBC-Jx-1.0-Jy0.0-Jz1.0-gamma1.0-dt0.05-init-all-plus.csv", -1.0, 0.0),
    ]
    for name, Jx, Jy in cases:
        ex = load(name)
        k = np.searchsorted(ex["t"], t_fit + 1e-9)
        t_out = ex["t"][:k]
        print(f"\n{name}\n  (s = H scale/sign, g = gamma factor; max |model - exact| for t <= {t_fit})")
        print(f"  {'s':>6} {'g':>5} " + "".join(f"{c:>9}" for c in ["Z0", "X0X1", "Z0Z1", "X0X2"]))
        for s in [1.0, -1.0, 0.5, 0.25]:
            for g in [0.5, 1.0, 2.0]:
                m = evolve(L, Jx, Jy, 1.0, 1.0, s, g, t_out)
                errs = [np.abs(m[c] - ex[c][:k]).max() for c in ["Z0Z0", "X0X1", "Z0Z1", "X0X2"]]
                flag = "  <== match" if max(errs) < 1e-3 else ""
                print(f"  {s:>6g} {g:>5g} " + "".join(f"{e:>9.4f}" for e in errs) + flag, flush=True)


if __name__ == "__main__":
    main()
