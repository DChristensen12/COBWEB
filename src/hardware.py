import os

import ffsim
import numpy as np
from pyscf import gto
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeKingston

from circuits import interaction_pairs, lucj_operator
from molecules import MOLECULES


def get_backend(name: str):

    # fake_kingston runs here on the laptop, with noise copied from the real chip
    if name == "fake_kingston":
        return FakeKingston()
    return QiskitRuntimeService().backend(name)


def lucj_circuit(op, norb: int, nelec: tuple[int, int]) -> QuantumCircuit:
    """Qiskit circuit that prepares Hartree-Fock and applies the LUCJ operator"""

    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(op), qubits)
    return circuit


def transpiled_circuit(name: str, n_reps: int, backend):
    """Measured LUCJ circuit for one molecule, laid out on the backend's qubits"""

    norb = gto.M(**MOLECULES[name], verbose=0).nao
    pairs_aa, pairs_ab = interaction_pairs(norb, "heavy-hex")

    # ffsim lays the alpha and beta chains side by side on the chip. If an
    # opposite-spin pair has no qubit to link it, it gets dropped from the list
    pass_manager, pairs_ab = ffsim.qiskit.generate_lucj_pass_manager(
        backend, norb, "heavy-hex", (pairs_aa, pairs_ab), optimization_level=3
    )
    op, _, norb, nelec = lucj_operator(name, n_reps, (pairs_aa, pairs_ab))
    circuit = lucj_circuit(op, norb, nelec)
    circuit.measure_all()
    return pass_manager.run(circuit), norb, nelec, pairs_ab


def split_bitstrings(bitstrings: list[str], norb: int):
    """Alpha and beta bit masks for each shot, in the order the shots came back"""

    # Qiskit writes qubit 0 on the right. The first norb qubits are alpha, so the
    # beta half is on the left
    ints = np.array([int(b, 2) for b in bitstrings], dtype=np.int64)
    return ints & ((1 << norb) - 1), ints >> norb


def main(
    results_dir: str,
    molecule: str,
    backend: str,
    n_shots: int,
    n_reps: int,
    submit: bool,
    job_id: str,
    overwrite: bool,
):

    out_path = f"{results_dir}/hardware/{molecule}.npz"
    if os.path.exists(out_path) and not overwrite:
        print(f"Skipping {out_path}, already done")
        return

    # The noisy simulator runs one shot at a time, so it gets far fewer than the QPU
    n_shots = n_shots or (1000 if backend.startswith("fake_") else 20000)
    device = get_backend(backend)
    isa, norb, nelec, pairs_ab = transpiled_circuit(molecule, n_reps, device)
    n_2q = isa.num_nonlocal_gates()
    print(f"{molecule} on {backend}: {2 * norb} qubits, opposite-spin pairs {pairs_ab}")
    print(f"Transpiled circuit has {n_2q} two-qubit gates and a depth of {isa.depth()}")

    # Anything but the fake backend spends QPU minutes, so it waits for --submit
    if job_id:
        job = QiskitRuntimeService().job(job_id)
    elif backend.startswith("fake_") or submit:
        job = SamplerV2(mode=device).run([isa], shots=n_shots)
        print(f"Job {job.job_id()} sent, waiting for {n_shots} shots")
    else:
        print("Nothing sent. Add --submit to run this on the QPU")
        return

    bitstrings = job.result()[0].data.meas.get_bitstrings()
    alpha, beta = split_bitstrings(bitstrings, norb)
    right = np.array(
        [
            bin(a).count("1") == nelec[0] and bin(b).count("1") == nelec[1]
            for a, b in zip(alpha, beta)
        ]
    )
    print(f"{right.mean():.1%} of {len(alpha)} shots have the right electron count")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        alpha=alpha,
        beta=beta,
        backend=backend,
        job_id=job.job_id(),
        n_two_qubit=n_2q,
        depth=isa.depth(),
        pairs_ab=np.array(pairs_ab),
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("-m", "--molecule", type=str, default="lih")
    parser.add_argument("-b", "--backend", type=str, default="fake_kingston")
    parser.add_argument("-n", "--n-shots", type=int, default=None)
    parser.add_argument("--n-reps", type=int, default=2)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--job-id", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    _args = parser.parse_args()
    main(**vars(_args))
