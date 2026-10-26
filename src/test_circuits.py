import ffsim
import numpy as np
from scipy.linalg import eigh

from circuits import configs_from_bitstrings, lucj_state
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
