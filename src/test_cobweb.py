import numpy as np
import pytest
from scipy.linalg import eigh
from hamiltonians import coupled_modes, dvr_kinetic, hubbard_chain
from utils import (
    discovery_curve,
    epstein_nesbet_order,
    epstein_nesbet_targets,
    fundamental_errors_by_size,
    fundamental_rows,
    reference_config,
    sample_order,
    sample_targets_order,
    subspace_energy,
    subspace_fundamentals,
    target_states,
    weight_order,
)
from vibrations import fundamentals, vci_hamiltonian


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


@pytest.fixture(scope="module")
def small_vci():
    # Three made-up modes with enough coupling that every eigenstate is mixed
    omega = np.array([800.0, 1500.0, 3000.0])
    phi3, phi4 = np.zeros((3,) * 3), np.zeros((3,) * 4)
    for i, j, k, v in [(0, 0, 1, -120.0), (1, 1, 2, 90.0), (0, 1, 2, 60.0)]:
        for perm in {(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)}:
            phi3[perm] = v
    phi4[0, 0, 0, 0] = 40.0
    H, labels = vci_hamiltonian(omega, phi3, phi4, 5)
    E, C = eigh(H)
    return H, labels, C, fundamentals(E, C, labels)


def test_fundamental_rows(small_vci):

    H, labels, _, _ = small_vci
    rows = fundamental_rows(labels)
    assert labels[rows[0]].sum() == 0
    assert np.array_equal(labels[rows[1:]], np.eye(3, dtype=int))


def test_subspace_fundamentals(small_vci):

    H, labels, _, nu = small_vci
    rows = fundamental_rows(labels)
    assert np.allclose(subspace_fundamentals(H, np.arange(len(H)), rows), nu)

    # Leave out mode 1's one-quantum row and its fundamental can't be found
    keep = np.delete(np.arange(len(H)), rows[2])
    assert np.isnan(subspace_fundamentals(H, keep, rows)).tolist() == [
        False,
        True,
        False,
    ]


def test_epstein_nesbet_targets_full_space(small_vci):

    # TASK 6 in the member guide. Grow the space all the way with
    # epstein_nesbet_targets and check the seeds come first, every row appears
    # once, and the fundamentals at full size match nu
    raise NotImplementedError("Task 6 in the member guide")


def test_sample_targets_order(small_vci):

    H, labels, C, _ = small_vci
    rows = fundamental_rows(labels)
    states = C[:, target_states(C, rows)] * 1j  # circuit states can be complex
    found, counts = sample_targets_order(states, 4001, np.random.default_rng(0))
    assert counts.sum() == 4000  # 1000 shots for each of the 4 states
    assert np.all(np.diff(counts) <= 0)
    assert set(rows) <= set(found.tolist())
    assert len(set(found.tolist())) == len(found)
