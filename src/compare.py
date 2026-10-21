import os

import numpy as np

from utils import (
    discovery_curve,
    epstein_nesbet_order,
    errors_by_size,
    reference_config,
    sample_order,
    weight_order,
)


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

    print(f"Sampling, {n_repeats} repeats of {n_shots} shots")
    rng = np.random.default_rng(seed)
    shots = np.unique(np.geomspace(10, 100 * n_shots, 30).astype(int))
    err_sample, n_found, discovered = [], [], []
    for i in range(n_repeats):
        found, _ = sample_order(c0, n_shots, rng)
        err_sample.append(errors_by_size(H, found, E[0], sizes))
        n_found.append(len(found))
        discovered.append(discovery_curve(c0, shots, rng))
    err_sample = np.stack(err_sample)  # repeats x sizes
    discovered = np.stack(discovered)  # repeats x shots
    print(f"Sampling found {np.mean(n_found):.0f} of {n_configs} configurations")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        sizes=sizes,
        err_weight=err_weight,
        err_en=err_en,
        err_sample=err_sample,
        n_found=np.array(n_found),
        shots=shots,
        discovered=discovered,
        n_shots=n_shots,
        n_configs=n_configs,
        units=units,
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
