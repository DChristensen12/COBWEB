from math import comb

import numpy as np
import pytest
from pyscf import gto
from pyscf.hessian import thermo
from scipy.linalg import eigh

from hamiltonians import dvr_kinetic
from vibrations import (
    EXPERIMENT,
    HARTREE_TO_CM,
    METHOD,
    VIBRATIONS,
    converged_quanta,
    equilibrium,
    fundamentals,
    ho_q_matrix,
    normal_modes,
    quartic_force_field,
    run_scf,
    vci_basis,
    vci_hamiltonian,
)


@pytest.fixture(scope="module")
def water():
    # The force field takes about a minute, so every test shares one
    return quartic_force_field("water")


def test_ho_q_matrix():

    # In the harmonic oscillator, <n|q^2|n> = n + 1/2
    q = ho_q_matrix(10)
    assert np.allclose(np.diag(q @ q)[:-1], np.arange(9) + 0.5)


def test_vci_basis_size():

    for n_modes, max_quanta in [(3, 6), (9, 4)]:
        assert len(vci_basis(n_modes, max_quanta)) == comb(
            n_modes + max_quanta, max_quanta
        )


def test_vci_matches_grid_for_one_mode():

    # One anharmonic mode, solved in the harmonic oscillator basis and on a grid
    omega, phi3, phi4 = (
        np.array([1500.0]),
        np.full((1,) * 3, -150.0),
        np.full((1,) * 4, 60.0),
    )
    H, _ = vci_hamiltonian(omega, phi3, phi4, 30)
    E_vci = eigh(H, eigvals_only=True)

    x = np.linspace(-6, 6, 201)
    V = 0.5 * omega[0] * x**2 + phi3[0, 0, 0] * x**3 / 6 + phi4[0, 0, 0, 0] * x**4 / 24
    E_grid = eigh(
        dvr_kinetic(len(x), x[1] - x[0], 1 / omega[0]) + np.diag(V), eigvals_only=True
    )
    assert np.allclose(E_vci[:4], E_grid[:4], rtol=0, atol=1e-6)


def test_water_harmonic_matches_pyscf():

    mol = equilibrium("water")
    mf = run_scf(mol)
    omega, modes, _ = normal_modes(mol, mf)
    ref = thermo.harmonic_analysis(mol, mf.Hessian().kernel())["freq_wavenumber"].real
    assert np.allclose(omega * HARTREE_TO_CM, ref, rtol=0, atol=1e-6)
    assert np.allclose(modes @ modes.T, np.eye(len(omega)))


def test_water_fundamentals_near_experiment(water):

    omega, phi3, phi4 = water
    H, labels = vci_hamiltonian(omega, phi3, phi4, 6)
    assert np.allclose(H, H.T)
    E, C = eigh(H)
    nu = np.sort(fundamentals(E, C, labels))
    assert np.all(np.abs(nu - EXPERIMENT["water"]) < 80)


def test_converged_quanta(water):

    omega, phi3, phi4 = water
    max_quanta, nu = converged_quanta(omega, phi3, phi4)
    H, labels = vci_hamiltonian(omega, phi3, phi4, max_quanta + 1)
    E, C = eigh(H)
    assert np.max(np.abs(fundamentals(E, C, labels) - nu)) < 1.0


def test_formic_acid_is_set_up():

    assert "formic_acid" in VIBRATIONS, "add formic acid to VIBRATIONS in vibrations.py"
    mol = gto.M(atom=VIBRATIONS["formic_acid"], basis=METHOD["basis"], verbose=0)
    assert sorted(mol.elements) == ["C", "H", "H", "O", "O"]

    # The geometry should be the trans conformer, with the hydroxyl hydrogen on
    # the same side as the carbonyl oxygen, a dihedral near zero
    xyz = mol.atom_coords()
    el = np.array(mol.elements)
    c = xyz[el == "C"][0]
    oxygens = xyz[el == "O"]
    carbonyl, hydroxyl = sorted(oxygens, key=lambda o: np.linalg.norm(o - c))
    h = min(xyz[el == "H"], key=lambda p: np.linalg.norm(p - hydroxyl))
    b0, b1, b2 = carbonyl - c, hydroxyl - c, h - hydroxyl
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - b0 @ b1 * b1, b2 - b2 @ b1 * b1
    dihedral = np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w))
    assert (
        abs(dihedral) < 30
    ), "use the trans conformer, O-H hydrogen next to the C=O oxygen"

    measured = EXPERIMENT["formic_acid"]
    assert len(measured) == 9 and list(measured) == sorted(measured)
    assert any(abs(m - 1306) < 5 for m in measured), "the O-H bend sits at 1306 cm-1"


def test_vci_harmonic_limit():

    # TODO with phi3 and phi4 all zero, the VCI matrix should be exactly diagonal,
    # with entries sum omega (n + 1/2). Build a small case and check both
    raise NotImplementedError("write this test")
