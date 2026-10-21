import os
from functools import reduce
from itertools import combinations, product

import numpy as np
from scipy.linalg import eigh

HARTREE_TO_CM = 219474.63


def hubbard_chain(n_sites: int, n_up: int, n_down: int, t: float = 1.0, U: float = 8.0):
    """Open-chain Hubbard Hamiltonian, one row per (up, down) occupation pair"""

    occs_up = [sum(1 << i for i in c) for c in combinations(range(n_sites), n_up)]
    occs_down = [sum(1 << i for i in c) for c in combinations(range(n_sites), n_down)]
    labels = np.array([(u, d) for u in occs_up for d in occs_down])
    index = {(u, d): k for k, (u, d) in enumerate(labels)}

    def hops(mask):
        return [
            mask ^ (1 << i) ^ (1 << (i + 1))
            for i in range(n_sites - 1)
            if (mask >> i & 1) != (mask >> (i + 1) & 1)
        ]

    # On an open chain with nearest-neighbour hops no electron sits between the two
    # sites, so the fermion sign is always +1 and we can skip that bookkeeping
    H = np.zeros((len(labels), len(labels)))
    for k, (u, d) in enumerate(labels):
        H[k, k] = U * bin(u & d).count("1")
        for u2 in hops(u):
            H[index[(u2, d)], k] -= t
        for d2 in hops(d):
            H[index[(u, d2)], k] -= t

    return H, labels


def dvr_kinetic(n_points: int, dx: float, mass: float):
    """Sinc-DVR kinetic energy matrix on a uniform grid"""
    # See Colbert and Miller, J. Chem. Phys. 96, 1982 (1992)
    # https://doi.org/10.1063/1.462100
    diff = np.subtract.outer(np.arange(n_points), np.arange(n_points))
    safe = np.where(diff == 0, 1, diff)
    T = np.where(diff == 0, np.pi**2 / 3, 2.0 * (-1.0) ** diff / safe**2)
    return T / (2 * mass * dx**2)


def morse_mode(D: float, a: float, mass: float, x: np.ndarray, n_levels: int):

    V = D * (1 - np.exp(-a * x)) ** 2
    E, psi = eigh(dvr_kinetic(len(x), x[1] - x[0], mass) + np.diag(V))
    S = psi[:, :n_levels].T @ np.diag(x**2) @ psi[:, :n_levels]
    return E[:n_levels], S


def coupled_modes(
    n_modes: int = 3,
    n_levels: int = 8,
    coupling: float = 20.0,
    mass: float = 918.0,
    D: float = 0.17,
    a: float = 1.03,
):
    """Morse modes coupled pairwise by coupling * x_i^2 x_j^2, returned in cm-1"""

    # Each mode is stiffer than the last so their frequencies differ, the usual case
    x = np.linspace(-1.2, 4.5, 300)
    modes = [
        morse_mode(D, a * (1 + 0.35 * i), mass, x, n_levels) for i in range(n_modes)
    ]
    eye = np.eye(n_levels)

    def embed(ops):
        return reduce(np.kron, [ops.get(i, eye) for i in range(n_modes)])

    H = sum(embed({i: np.diag(E)}) for i, (E, _) in enumerate(modes))
    for i, j in combinations(range(n_modes), 2):
        H = H + coupling * embed({i: modes[i][1], j: modes[j][1]})

    # kron puts mode 0 on the slowest index, the same order product gives
    labels = np.array(list(product(range(n_levels), repeat=n_modes)))
    return H * HARTREE_TO_CM, labels


def main(
    results_dir: str,
    system: str,
    n_sites: int,
    U: float,
    n_modes: int,
    n_levels: int,
    coupling: float,
    overwrite: bool,
):

    out_path = f"{results_dir}/hamiltonians/{system}.npz"
    if os.path.exists(out_path) and not overwrite:
        print(f"Skipping {out_path}, already done")
        return

    if system == "hubbard":
        H, labels = hubbard_chain(n_sites, n_sites // 2, n_sites // 2, U=U)
        units = "t"
    elif system == "modes":
        H, labels = coupled_modes(n_modes, n_levels, coupling)
        units = "cm-1"
    else:
        raise ValueError(f"unknown system {system}, pass hubbard or modes")

    assert np.allclose(H, H.T), "hamiltonian is not symmetric"
    print(f"{system}: H has a shape of {H.shape}, labels {labels.shape}")

    E, C = eigh(H)
    print(f"Ground state energy {E[0]:.6f} {units}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, H=H, labels=labels, E=E, c0=C[:, 0], units=units)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("-s", "--system", type=str, default="hubbard")
    parser.add_argument("--n-sites", type=int, default=6)
    parser.add_argument("-U", type=float, default=8.0)
    parser.add_argument("--n-modes", type=int, default=3)
    parser.add_argument("--n-levels", type=int, default=8)
    parser.add_argument("--coupling", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    _args = parser.parse_args()
    main(**vars(_args))
