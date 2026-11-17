import os

import ffsim
import numpy as np
from pyscf import cc, gto, scf

from molecules import MOLECULES


def interaction_pairs(norb: int, pattern: str):
    """Orbital pairs the LUCJ circuit keeps, (alpha-alpha, alpha-beta), for a qubit layout"""

    # Same-spin neighbours sit next to each other in both layouts. An opposite-spin
    # pair needs a qubit linking the alpha and beta chains, which a square grid has
    # at every orbital and IBM's heavy-hex Heron chips have at every fourth, see
    # https://eu-de.quantum.cloud.ibm.com/docs/en/tutorials/sample-based-quantum-diagonalization
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    if pattern == "square":
        return pairs_aa, [(p, p) for p in range(norb)]
    if pattern == "heavy-hex":
        return pairs_aa, [(p, p) for p in range(0, norb, 4)]
    raise ValueError(f"unknown pattern {pattern}, use square or heavy-hex")


def lucj_operator(name: str, n_reps: int = 2, pairs=None):
    """LUCJ operator with parameters taken straight from CCSD, no optimization"""

    mol = gto.M(**MOLECULES[name], verbose=0)
    mf = scf.RHF(mol).run()
    ccsd = cc.CCSD(mf).run()
    assert ccsd.converged, f"CCSD did not converge for {name}"
    op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        ccsd.t2, t1=ccsd.t1, n_reps=n_reps, interaction_pairs=pairs
    )
    return op, mf, mol.nao, mol.nelec


def lucj_state(name: str, n_reps: int = 2, local: bool = True, pattern="square"):
    """LUCJ state vector and its energy, with local pairs only unless local is False"""

    norb = gto.M(**MOLECULES[name], verbose=0).nao
    pairs = interaction_pairs(norb, pattern) if local else None  # None keeps every pair
    op, mf, norb, nelec = lucj_operator(name, n_reps, pairs)
    vec = ffsim.apply_unitary(
        ffsim.hartree_fock_state(norb, nelec), op, norb=norb, nelec=nelec
    )

    # Energy with ffsim's own Hamiltonian, to check against ours later
    ham = ffsim.MolecularData.from_scf(mf).hamiltonian
    e_ffsim = np.real(np.vdot(vec, ffsim.linear_operator(ham, norb, nelec) @ vec))
    return vec, e_ffsim, norb, nelec


def configs_from_bitstrings(alpha, beta, labels: np.ndarray):
    """Rows of H for measured (alpha, beta) bit masks, most frequent first"""

    # Noise on hardware produces bitstrings with the wrong electron count. Those
    # are not rows of H, so we drop them here instead of recovering them
    index = {(int(a), int(b)): k for k, (a, b) in enumerate(labels)}
    rows = [index[(a, b)] for a, b in zip(alpha, beta) if (a, b) in index]
    found, counts = np.unique(rows, return_counts=True)
    order = np.argsort(counts, kind="stable")[::-1]
    return found[order], counts[order]


def main(results_dir: str, molecule: str, n_reps: int, pattern: str, overwrite: bool):

    in_path = f"{results_dir}/hamiltonians/{molecule}.npz"
    suffix = "" if pattern == "square" else "_" + pattern.replace("-", "_")
    out_path = f"{results_dir}/circuits/{molecule}{suffix}.npz"
    assert os.path.exists(in_path), f"no hamiltonian at {in_path}, run make molecules"
    if os.path.exists(out_path) and not overwrite:
        print(f"Skipping {out_path}, already done")
        return

    data = np.load(in_path)
    H, E = data["H"], data["E"]
    vec, e_ffsim, norb, nelec = lucj_state(molecule, n_reps, pattern=pattern)
    assert len(vec) == len(H), "circuit state and hamiltonian disagree on size"

    # If the determinant order or orbital signs differed between ffsim and our H,
    # these two energies would not match
    e_lucj = np.real(np.vdot(vec, H @ vec))
    assert np.isclose(
        e_lucj, e_ffsim, rtol=0, atol=1e-8
    ), f"energies differ: {e_lucj} vs {e_ffsim}"
    print(f"{molecule}: {2 * norb} qubits, {n_reps} LUCJ layers, {pattern} pairs")
    print(f"LUCJ {e_lucj:.6f}, full CI {E[0]:.6f} hartree")
    print(f"Largest single weight {np.max(np.abs(vec) ** 2):.3f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, c=vec, e_lucj=e_lucj, n_reps=n_reps, n_qubits=2 * norb)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("-m", "--molecule", type=str, default="h4")
    parser.add_argument("--n-reps", type=int, default=2)
    parser.add_argument("-p", "--pattern", type=str, default="square")
    parser.add_argument("--overwrite", action="store_true")
    _args = parser.parse_args()
    main(**vars(_args))
