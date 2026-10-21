import numpy as np
from pyscf import fci, gto, scf
from scipy.linalg import eigh

from molecules import MOLECULES, determinant_labels, fci_hamiltonian, integrals
from utils import reference_config


def build(name):

    h1, h2, ecore, norb, nelec, e_hf = integrals(name)
    H = fci_hamiltonian(h1, h2, ecore, norb, nelec)
    return H, determinant_labels(norb, nelec), norb, nelec, e_hf


def pyscf_fci_energy(name):

    mf = scf.RHF(gto.M(**MOLECULES[name], verbose=0)).run()
    return fci.FCI(mf).kernel()[0]


def test_hamiltonians_match_pyscf_fci():

    for name in ["h4", "lih"]:
        h1, h2, ecore, norb, nelec, _ = integrals(name)
        H = fci_hamiltonian(h1, h2, ecore, norb, nelec)
        assert np.allclose(H, H.T)
        assert np.isclose(eigh(H, eigvals_only=True)[0], pyscf_fci_energy(name))


def test_water_matches_pyscf_fci():

    # Water in STO-3G has 7 orbitals and 5 electrons of each spin, so 21 x 21
    assert "h2o" in MOLECULES, "add water to MOLECULES in molecules.py"
    h1, h2, ecore, norb, nelec, _ = integrals("h2o")
    H = fci_hamiltonian(h1, h2, ecore, norb, nelec)
    assert len(H) == 441
    assert np.isclose(eigh(H, eigvals_only=True)[0], pyscf_fci_energy("h2o"))


def test_labels_are_in_fci_order():

    # With canonical RHF orbitals the Hartree-Fock determinant's diagonal entry is
    # the Hartree-Fock energy, which only works out if labels and rows line up
    for name in ["h4", "lih"]:
        H, labels, norb, nelec, e_hf = build(name)
        assert labels.shape == (len(H), 2)
        assert all(bin(a).count("1") == nelec[0] for a in labels[:, 0])
        assert all(bin(b).count("1") == nelec[1] for b in labels[:, 1])

        hf = (1 << nelec[0]) - 1, (1 << nelec[1]) - 1  # lowest orbitals filled
        k = np.where((labels[:, 0] == hf[0]) & (labels[:, 1] == hf[1]))[0][0]
        assert np.isclose(H[k, k], e_hf)


def test_reference_config_is_hartree_fock():

    # TODO the cheapest determinant on the diagonal should be Hartree-Fock, the usual
    # starting point for selected CI. For h4 and lih, build H with integrals and
    # fci_hamiltonian, then check the diagonal entry at reference_config(H) equals
    # the Hartree-Fock energy that integrals returns
    raise NotImplementedError("write this test")
