from circuits import interaction_pairs, lucj_operator, lucj_state
from hardware import lucj_circuit, split_bitstrings
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
import ffsim 
import numpy as np


def test_circuit_matches_lucj_state():
    names = ["lih", "h2o"]
    for name in names:
        acutal_state, _, norb, _ = lucj_state(name)
        op, _, norb, nelec = lucj_operator(name, pairs=interaction_pairs(norb, pattern="square"))

        qc = lucj_circuit(op=op, norb=norb, nelec=nelec)
        sv = Statevector.from_instruction(qc)
        sv_lucj = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            acutal_state, norb=norb, nelec=nelec
        )

        fidelity = np.abs(np.vdot(sv.data, sv_lucj.data)) ** 2
        tr_distance = np.sqrt(max(0.0, 1.0 - fidelity))

        assert np.isclose(tr_distance, 0.0, atol=1e-6)


def test_split_bitstrings():
    norb = 6
    bitstrings = ["011000000011", "011001000011"]
    alpha, beta = split_bitstrings(bitstrings=bitstrings, norb=norb)
    solns = [["0b000011"] * 2, ["0b011000", "0b011001"]]
    for i in range(len(solns)):
        assert f"0b{alpha[i]:0{norb}b}" == solns[0][i]
        assert f"0b{beta[i]:0{norb}b}" == solns[1][i]


