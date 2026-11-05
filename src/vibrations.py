import logging
import os
from itertools import combinations_with_replacement, permutations, product
from math import factorial

import numpy as np
from pyscf import dft, gto
from pyscf.data import nist
from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian import thermo
from scipy.linalg import eigh

HARTREE_TO_CM = nist.HARTREE2WAVENUMBER
AMU_TO_ME = nist.AMU2AU
ANGSTROM_TO_BOHR = 1 / nist.BOHR
METHOD = dict(xc="b3lyp", basis="6-31+g(d,p)")

# Starting geometries in angstrom, the optimizer takes them the rest of the way
VIBRATIONS = {
    "water": "O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467",
    # TODO add formic acid as "formic_acid". A rough geometry is fine as long as it is
    # the trans conformer, with the O-H hydrogen on the same side as the C=O oxygen
}

# VCI basis size per molecule. Formic acid stops at 4 because its quartic force
# field has a hole: the torsion couples so strongly to the O-H stretch that the
# potential turns negative at q of about 2.5, where DFT is still 1700 cm-1 uphill.
# Bigger bases reach into it, and the fundamentals stop converging
MAX_QUANTA = {"water": 6, "formic_acid": 4}

# Measured gas-phase fundamentals in cm-1, lowest to highest
EXPERIMENT = {
    "water": [1595, 3657, 3756],
    # TODO add the nine formic acid fundamentals from Table I of Nejad and Sibert,
    # J. Chem. Phys. 154, 064301 (2021), https://doi.org/10.1063/5.0039237. The older
    # NIST list puts the O-H bend at 1229, which turned out to be a torsion overtone
}


def run_scf(mol):

    mf = dft.RKS(mol, xc=METHOD["xc"])
    mf.grids.level = 4  # finite differences of Hessians need a smooth grid
    mf.conv_tol = 1e-11
    return mf.run()


def equilibrium(name: str):
    """Optimized geometry for one molecule at the level in METHOD"""

    mol = gto.M(atom=VIBRATIONS[name], basis=METHOD["basis"], verbose=0)
    logging.disable(logging.CRITICAL)  # geomeTRIC logs every step
    mol = optimize(run_scf(mol), maxsteps=100)
    logging.disable(logging.NOTSET)
    return mol


def normal_modes(mol, mf):
    """Harmonic frequencies in hartree and mass-weighted mode vectors, one per row"""

    result = thermo.harmonic_analysis(mol, mf.Hessian().kernel())
    assert (
        result["freq_error"] == 0
    ), "imaginary frequency, the geometry is not a minimum"
    mass = mol.atom_mass_list(isotope_avg=True)
    modes = result["norm_mode"] * np.sqrt(mass)[None, :, None]
    omega = result["freq_au"].real / np.sqrt(AMU_TO_ME)
    return omega, modes.reshape(len(omega), -1), mass


def hessian_in_modes(mol, coords, modes, omega, mass):
    """Second derivatives along each pair of dimensionless normal coordinates, in hartree"""

    displaced = mol.copy()
    displaced.set_geom_(coords, unit="Bohr")
    hess = run_scf(displaced).Hessian().kernel()
    hess = hess.transpose(0, 2, 1, 3).reshape(3 * mol.natm, 3 * mol.natm)
    w = 1 / np.sqrt(np.repeat(mass, 3) * AMU_TO_ME)
    return modes @ (hess * np.outer(w, w)) @ modes.T / np.sqrt(np.outer(omega, omega))


def quartic_force_field(name: str, step: float = 0.025):
    """Harmonic frequencies, cubic and semi-diagonal quartic force constants, in cm-1"""
    # Third derivatives from first differences of Hessians along each mode, and
    # fourth derivatives of the form ijkk from second differences, as in
    # Barone, J. Chem. Phys. 122, 014108 (2005), with his step of 0.025 angstrom amu^1/2

    mol = equilibrium(name)
    omega, modes, mass = normal_modes(mol, run_scf(mol))
    coords = mol.atom_coords()
    n = len(omega)
    K0 = hessian_in_modes(mol, coords, modes, omega, mass)

    cubic, quartic = {}, {}
    for k in range(n):
        step_q = step * ANGSTROM_TO_BOHR * np.sqrt(AMU_TO_ME * omega[k])
        dx = (modes[k] / np.sqrt(np.repeat(mass, 3) * AMU_TO_ME)).reshape(-1, 3)
        dx = dx * step_q / np.sqrt(omega[k])
        Kp = hessian_in_modes(mol, coords + dx, modes, omega, mass)
        Km = hessian_in_modes(mol, coords - dx, modes, omega, mass)
        first = (Kp - Km) / (2 * step_q)
        second = (Kp + Km - 2 * K0) / step_q**2
        for i, j in product(range(n), repeat=2):
            cubic.setdefault(tuple(sorted((i, j, k))), []).append(first[i, j])
            quartic.setdefault(tuple(sorted((i, j, k, k))), []).append(second[i, j])

    # Most constants come out of more than one displacement, so we average them
    phi3, phi4 = np.zeros((n,) * 3), np.zeros((n,) * 4)
    for idx, values in cubic.items():
        for perm in set(permutations(idx)):
            phi3[perm] = np.mean(values)
    for idx, values in quartic.items():
        for perm in set(permutations(idx)):
            phi4[perm] = np.mean(values)

    return omega * HARTREE_TO_CM, phi3 * HARTREE_TO_CM, phi4 * HARTREE_TO_CM


def ho_q_matrix(size: int) -> np.ndarray:
    """Dimensionless position q in the first size harmonic oscillator states"""

    n = np.arange(1, size)
    return np.diag(np.sqrt(n / 2), 1) + np.diag(np.sqrt(n / 2), -1)


def vci_basis(n_modes: int, max_quanta: int) -> np.ndarray:
    """Every product state with at most max_quanta quanta in total, one row per state"""

    labels = []
    for total in range(max_quanta + 1):
        for modes in combinations_with_replacement(range(n_modes), total):
            labels.append(np.bincount(modes, minlength=n_modes))
    return np.array(labels).reshape(-1, n_modes)


def vci_hamiltonian(omega, phi3, phi4, max_quanta: int):
    """VCI matrix in the harmonic oscillator product basis, in cm-1"""

    n = len(omega)
    labels = vci_basis(n, max_quanta)
    H = np.diag((labels + 0.5) @ omega)

    # Powers of q come from a bigger basis, so truncation does not touch the ones we keep
    q = ho_q_matrix(max_quanta + 5)
    keep = slice(0, max_quanta + 1)
    power = {c: np.linalg.matrix_power(q, c)[keep, keep] for c in range(1, 5)}

    base = (max_quanta + 1) ** np.arange(n)
    keys = labels @ base
    order = np.argsort(keys)

    # Summing phi_S / prod(c!) times prod(q^c) over every multiset S of modes is the
    # same as the full Taylor sum with its 1/3! and 1/4!, once permutations are counted
    terms = [(s, phi3[s]) for s in combinations_with_replacement(range(n), 3)]
    terms += [(s, phi4[s]) for s in combinations_with_replacement(range(n), 4)]
    for s, phi in terms:
        if abs(phi) < 1e-10:
            continue
        mode, count = np.unique(s, return_counts=True)
        coeff = phi / np.prod([factorial(c) for c in count])
        for shift in product(*[range(-c, c + 1, 2) for c in count]):
            target = labels.copy()
            target[:, mode] += shift
            ok = (target >= 0).all(1) & (target.sum(1) <= max_quanta)
            rows = order[np.searchsorted(keys, target[ok] @ base, sorter=order)]
            value = np.prod(
                [power[c][target[ok, m], labels[ok, m]] for m, c in zip(mode, count)], 0
            )
            H[rows, np.where(ok)[0]] += coeff * value

    return H, labels


def fundamentals(E: np.ndarray, C: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fundamental of each mode, from the state that looks most like one quantum in it"""

    freq = []
    for i in range(labels.shape[1]):
        row = np.where((labels.sum(1) == 1) & (labels[:, i] == 1))[0][0]
        state = np.argmax(C[row, 1:] ** 2) + 1
        freq.append(E[state] - E[0])
    return np.array(freq)


def converged_quanta(
    omega, phi3, phi4, tol: float = 1.0, start: int = 2, stop: int = 6
):
    """Smallest max_quanta whose fundamentals move by less than tol when it goes up by one"""

    # TODO build the VCI matrix at max_quanta and at max_quanta + 1, starting from
    # start, and return (max_quanta, fundamentals) for the first pair that agrees to
    # within tol. Raise an error if nothing agrees by stop
    raise NotImplementedError("converged_quanta is a week 4 contribution spot")


def main(results_dir: str, molecule: str, max_quanta: int, overwrite: bool):

    max_quanta = max_quanta or MAX_QUANTA[molecule]
    ff_path = f"{results_dir}/forcefields/{molecule}.npz"
    out_path = f"{results_dir}/hamiltonians/{molecule}.npz"
    if os.path.exists(out_path) and not overwrite:
        print(f"Skipping {out_path}, already done")
        return

    # The force field is the slow part, so we keep it even when the matrix is redone
    if not os.path.exists(ff_path):
        print(f"{molecule}: computing the quartic force field, this takes a while")
        omega, phi3, phi4 = quartic_force_field(molecule)
        os.makedirs(os.path.dirname(ff_path), exist_ok=True)
        np.savez(ff_path, omega=omega, phi3=phi3, phi4=phi4)
    ff = np.load(ff_path)
    omega, phi3, phi4 = ff["omega"], ff["phi3"], ff["phi4"]

    H, labels = vci_hamiltonian(omega, phi3, phi4, max_quanta)
    assert np.allclose(H, H.T), "hamiltonian is not symmetric"
    print(f"{molecule}: {len(omega)} modes, H has a shape of {H.shape}")

    E, C = eigh(H)
    nu = fundamentals(E, C, labels)
    print(f"Harmonic     {np.round(np.sort(omega))}")
    print(f"Fundamentals {np.round(np.sort(nu))}")
    if molecule in EXPERIMENT:
        print(f"Measured     {np.array(EXPERIMENT[molecule])}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path, H=H, labels=labels, E=E, c0=C[:, 0], units="cm-1", omega=omega, nu=nu
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("-m", "--molecule", type=str, default="water")
    parser.add_argument("-q", "--max-quanta", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    _args = parser.parse_args()
    main(**vars(_args))
