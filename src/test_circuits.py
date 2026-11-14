import ffsim
import numpy as np
from scipy.linalg import eigh

from circuits import configs_from_bitstrings, interaction_pairs, lucj_state
from molecules import determinant_labels, fci_hamiltonian, integrals
from utils import sample_order


def test_lucj_energy_matches_ffsim():

    # Our H and ffsim's Hamiltonian only give the same energy if the determinant
    # order and the orbital signs line up, which compare.py relies on
    for name in ["lih", "h2o"]:
        h1, h2, ecore, norb, nelec, _ = integrals(name)
        H = fci_hamiltonian(h1, h2, ecore, norb, nelec)
        vec, e_ffsim, _, _ = lucj_state(name)
        assert len(vec) == len(H)
        assert np.isclose(np.real(np.vdot(vec, H @ vec)), e_ffsim, rtol=0, atol=1e-8)


def test_lucj_is_variational():

    for name in ["lih", "h2o"]:
        h1, h2, ecore, norb, nelec, _ = integrals(name)
        E0 = eigh(fci_hamiltonian(h1, h2, ecore, norb, nelec), eigvals_only=True)[0]
        vec, e_lucj, _, _ = lucj_state(name)
        assert np.isclose(np.vdot(vec, vec).real, 1.0)
        assert e_lucj >= E0 - 1e-9


def test_configs_from_bitstrings():

    vec, _, norb, nelec = lucj_state("lih")
    labels = determinant_labels(norb, nelec)
    alpha, beta = ffsim.sample_state_vector(
        vec,
        norb=norb,
        nelec=nelec,
        shots=2000,
        concatenate=False,
        bitstring_type=ffsim.BitstringType.INT,
        seed=0,
    )
    found, counts = configs_from_bitstrings(alpha, beta, labels)
    assert counts.sum() == 2000
    assert found[0] == 0  # Hartree-Fock is row 0 and the most common sample

    # A bitstring with three alpha electrons is not a row of H and gets dropped
    found2, counts2 = configs_from_bitstrings(
        np.append(alpha, 0b111), np.append(beta, 0b11), labels
    )
    assert counts2.sum() == 2000 and np.array_equal(found2, found)


def test_sampling_handles_complex_amplitudes():

    vec, _, _, _ = lucj_state("lih")
    a = sample_order(vec, 5000, np.random.default_rng(3))
    b = sample_order(np.abs(vec), 5000, np.random.default_rng(3))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_heavy_hex_pairs():

    # Heavy-hex keeps the same-spin chain but links alpha and beta only every
    # fourth orbital
    square_aa, square_ab = interaction_pairs(8, "square")
    aa, ab = interaction_pairs(8, "heavy-hex")
    assert aa == square_aa == [(p, p + 1) for p in range(7)]
    assert ab == [(0, 0), (4, 4)]
    assert interaction_pairs(6, "heavy-hex")[1] == [(0, 0), (4, 4)]
    assert interaction_pairs(3, "heavy-hex")[1] == [(0, 0)]


def test_heavy_hex_is_variational():

    h1, h2, ecore, norb, nelec, _ = integrals("lih")
    E0 = eigh(fci_hamiltonian(h1, h2, ecore, norb, nelec), eigvals_only=True)[0]
    _, e_square, _, _ = lucj_state("lih")
    _, e_hh, _, _ = lucj_state("lih", pattern="heavy-hex")
    assert E0 - 1e-9 <= e_square and E0 - 1e-9 <= e_hh
