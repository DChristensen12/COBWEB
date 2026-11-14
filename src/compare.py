import os

import numpy as np
from scipy.linalg import eigh

from circuits import configs_from_bitstrings
from utils import (
    discovery_curve,
    epstein_nesbet_order,
    epstein_nesbet_targets,
    errors_by_size,
    fundamental_errors_by_size,
    fundamental_rows,
    reference_config,
    sample_order,
    sample_targets_order,
    target_states,
    weight_order,
)


def sample_errors(H, c, E0, sizes, shots, n_shots, n_repeats, rng):
    """Errors, configurations found and discovery curves over repeated sampling runs"""

    errors, n_found, discovered = [], [], []
    for _ in range(n_repeats):
        found, _ = sample_order(c, n_shots, rng)
        errors.append(errors_by_size(H, found, E0, sizes))
        n_found.append(len(found))
        discovered.append(discovery_curve(c, shots, rng))
    # repeats x sizes, repeats, repeats x shots
    return np.stack(errors), np.array(n_found), np.stack(discovered)


def main(
    results_dir: str,
    system: str,
    n_shots: int,
    n_repeats: int,
    batch: int,
    seed: int,
    overwrite: bool,
):

    in_path = f"{results_dir}/hamiltonians/{system}.npz"
    out_path = f"{results_dir}/compare/{system}.npz"
    assert os.path.exists(
        in_path
    ), f"no hamiltonian at {in_path}, run make hamiltonians"
    if os.path.exists(out_path) and not overwrite:
        print(f"Skipping {out_path}, already done")
        return

    data = np.load(in_path)
    H, E, c0, units = data["H"], data["E"], data["c0"], str(data["units"])
    n_configs = len(H)
    print(f"{system}: {n_configs} configurations, energies in {units}")

    # Every selector returns an ordering, and the subspace of size k is its first k
    # entries. Perturbative growth is deterministic, so the first k of a run to the
    # full space match a run stopped at k, which lets us grow once
    # log spaced so the small subspaces, where sampling lives, are resolved
    sizes = np.unique(np.geomspace(1, n_configs, 40).astype(int))
    seed_config = [reference_config(H)]

    print("Ranking by exact weight")
    err_weight = errors_by_size(H, weight_order(c0), E[0], sizes)

    print("Growing with Epstein-Nesbet")
    err_en = errors_by_size(
        H, epstein_nesbet_order(H, seed_config, n_configs, batch), E[0], sizes
    )

    rng = np.random.default_rng(seed)
    shots = np.unique(np.geomspace(10, 100 * n_shots, 30).astype(int))

    print(f"Sampling the exact state, {n_repeats} repeats of {n_shots} shots")
    err_sample, n_found, discovered = sample_errors(
        H, c0, E[0], sizes, shots, n_shots, n_repeats, rng
    )
    print(f"Found {np.mean(n_found):.0f} of {n_configs} configurations")

    # Molecules also get circuit states from circuits.py, sampled the same way
    extra = {}
    for key, suffix in [("circuit", ""), ("circuit_hh", "_heavy_hex")]:
        circuit_path = f"{results_dir}/circuits/{system}{suffix}.npz"
        if not os.path.exists(circuit_path):
            continue
        c_circuit = np.load(circuit_path)["c"]
        print(f"Sampling {circuit_path}, {n_repeats} repeats of {n_shots} shots")
        err, found, disc = sample_errors(
            H, c_circuit, E[0], sizes, shots, n_shots, n_repeats, rng
        )
        print(f"Found {np.mean(found):.0f} of {n_configs} configurations")
        extra.update(
            {f"err_{key}": err, f"n_found_{key}": found, f"discovered_{key}": disc}
        )

    # A hardware or noisy simulator run is a single set of shots, kept in shot order
    hardware_path = f"{results_dir}/hardware/{system}.npz"
    if os.path.exists(hardware_path):
        hw = np.load(hardware_path)
        alpha, beta, labels = hw["alpha"], hw["beta"], data["labels"]
        found, counts = configs_from_bitstrings(alpha, beta, labels)
        seen = [
            len(configs_from_bitstrings(alpha[:s], beta[:s], labels)[0]) for s in shots
        ]
        print(
            f"{hw['backend']}: {counts.sum()} of {len(alpha)} shots usable, found {len(found)}"
        )
        extra.update(
            err_hardware=errors_by_size(H, found, E[0], sizes),
            discovered_hardware=np.where(shots <= len(alpha), seen, np.nan),
            usable_hardware=counts.sum() / len(alpha),
            backend_hardware=str(hw["backend"]),
        )

    # Vibrations also get the fundamentals, the band positions an IR spectrum shows.
    # Perturbation theory and sampling both target the ground state and every
    # one-quantum state at once, starting from the same seed rows
    if "nu" in data.files:
        rows = fundamental_rows(data["labels"])
        _, C = eigh(H)
        states = C[:, target_states(C, rows)]  # configurations x targets
        print(f"Growing with Epstein-Nesbet for {len(rows)} target states")
        order = epstein_nesbet_targets(H, list(rows), n_configs, batch)
        extra["fund_en"] = fundamental_errors_by_size(H, order, rows, data["nu"], sizes)
        print(f"Sampling the target states, {n_repeats} repeats of {n_shots} shots")
        fund = []
        for _ in range(n_repeats):
            found, _ = sample_targets_order(states, n_shots, rng)
            fund.append(fundamental_errors_by_size(H, found, rows, data["nu"], sizes))
        extra["fund_sample"] = np.stack(fund)  # repeats x sizes

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        sizes=sizes,
        err_weight=err_weight,
        err_en=err_en,
        err_sample=err_sample,
        n_found=n_found,
        shots=shots,
        discovered=discovered,
        n_shots=n_shots,
        n_configs=n_configs,
        units=units,
        **extra,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("-s", "--system", type=str, default="hubbard")
    parser.add_argument("-n", "--n-shots", type=int, default=20000)
    parser.add_argument("-r", "--n-repeats", type=int, default=20)
    parser.add_argument("-b", "--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    _args = parser.parse_args()
    main(**vars(_args))
