import numpy as np
import pytest
from scipy.linalg import eigh

from hamiltonians import coupled_modes, dvr_kinetic, hubbard_chain
from utils import (
    discovery_curve,
    epstein_nesbet_order,
    reference_config,
    sample_order,
    subspace_energy,
    weight_order,
)


def test_dvr_matches_exact_morse_levels():

    mass, D, a = 918.0, 0.17, 1.03
    omega = a * np.sqrt(2 * D / mass)
    x = np.linspace(-1.2, 4.5, 300)
    V = D * (1 - np.exp(-a * x)) ** 2
    E = eigh(dvr_kinetic(len(x), x[1] - x[0], mass) + np.diag(V), eigvals_only=True)

    # Morse levels are known in closed form, E_n = w(n + 1/2) - w^2 (n + 1/2)^2 / 4D
    n = np.arange(8)
    exact = omega * (n + 0.5) - omega**2 / (4 * D) * (n + 0.5) ** 2
    assert np.allclose(E[:8], exact, rtol=0, atol=1e-10)


def test_hubbard_free_fermions():

    # At U = 0 the ground state fills the lowest single-particle levels of the chain,
    # which for an open chain are -2t cos(k pi / (L + 1))
    H, _ = hubbard_chain(6, 3, 3, U=0.0)
    eps = np.sort(-2 * np.cos(np.arange(1, 7) * np.pi / 7))
    assert np.isclose(
        eigh(H, eigvals_only=True)[0], 2 * eps[:3].sum(), rtol=0, atol=1e-10
    )


def test_hubbard_matches_pyscf():

    fci = pytest.importorskip("pyscf.fci")

    # Same h1 and eri construction as PySCF's own Hubbard example, without the
    # periodic bond, https://github.com/pyscf/pyscf/blob/master/examples/scf/40-customizing_hamiltonian.py
    n, U = 6, 8.0
    h1 = np.zeros((n, n))
    for i in range(n - 1):
        h1[i, i + 1] = h1[i + 1, i] = -1.0
    eri = np.zeros((n, n, n, n))
    for i in range(n):
        eri[i, i, i, i] = U
    e_pyscf, _ = fci.direct_spin1.kernel(h1, eri, n, (3, 3))

    H, _ = hubbard_chain(n, 3, 3, U=U)
    assert np.isclose(eigh(H, eigvals_only=True)[0], e_pyscf, rtol=0, atol=1e-9)


def test_hamiltonians_are_symmetric():

    for H, _ in [hubbard_chain(6, 3, 3), coupled_modes(3, 6)]:
        assert np.allclose(H, H.T)


def test_mode_labels_match_kron_order():

    # With no coupling the diagonal is a sum of single-mode levels, so the cheapest
    # configuration has to be all modes in their ground state
    H, labels = coupled_modes(3, 4, coupling=0.0)
    assert np.allclose(H, np.diag(np.diag(H)))
    assert tuple(labels[np.argmin(np.diag(H))]) == (0, 0, 0)


def test_full_subspace_is_exact():

    H, _ = hubbard_chain(6, 3, 3)
    E = eigh(H, eigvals_only=True)
    assert np.isclose(subspace_energy(H, np.arange(len(H))), E[0], rtol=0, atol=1e-10)


def test_subspace_energy_is_variational():

    # Keeping fewer configurations can never give an energy below the exact one
    H, _ = coupled_modes(3, 6)
    E, C = eigh(H)
    order = weight_order(C[:, 0])
    energies = [subspace_energy(H, order[:k]) for k in range(1, len(H) + 1, 20)]
    assert np.all(np.array(energies) >= E[0] - 1e-9)
    assert np.all(np.diff(energies) <= 1e-9)  # adding configurations only helps


def test_epstein_nesbet_is_a_full_permutation():

    H, _ = hubbard_chain(6, 3, 3)
    order = epstein_nesbet_order(H, [reference_config(H)], len(H))
    assert sorted(order.tolist()) == list(range(len(H)))


def test_epstein_nesbet_prefix_matches_shorter_run():

    # compare.py grows once to the full space and slices, which relies on this
    H, _ = hubbard_chain(6, 3, 3)
    seed = [reference_config(H)]
    full = epstein_nesbet_order(H, seed, len(H))
    for k in [1, 7, 50, 133]:
        assert np.array_equal(epstein_nesbet_order(H, seed, k), full[:k])


def test_sampling_returns_unique_valid_configurations():

    H, _ = hubbard_chain(6, 3, 3)
    E, C = eigh(H)
    found, counts = sample_order(C[:, 0], 5000, np.random.default_rng(0))
    assert len(found) == len(set(found.tolist()))
    assert found.min() >= 0 and found.max() < len(H)
    assert counts.sum() == 5000
    assert np.all(np.diff(counts) <= 0)  # most frequent first


def test_discovery_curve_matches_brute_force():

    c0 = np.sqrt(np.array([0.5, 0.3, 0.15, 0.05]))
    shots = np.array([1, 2, 5, 10, 100])
    fast = discovery_curve(c0, shots, np.random.default_rng(1))

    draws = np.random.default_rng(1).choice(4, size=100, p=c0**2 / np.sum(c0**2))
    brute = [len(np.unique(draws[:n])) for n in shots]
    assert np.array_equal(fast, brute)


def test_reference_config_breaks_ties_physically():

    # At large U every singly filled chain ties on the diagonal. The alternating
    # (Neel) arrangement has the most antiparallel neighbours, so it should win
    H, labels = hubbard_chain(6, 3, 3, U=8.0)
    up, down = labels[reference_config(H)]
    assert {f"{up:06b}", f"{down:06b}"} == {"010101", "101010"}

    H, labels = coupled_modes(3, 4)
    assert tuple(labels[reference_config(H)]) == (0, 0, 0)
