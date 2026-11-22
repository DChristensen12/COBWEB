import numpy as np
from scipy.linalg import eigh


def ground_state(H: np.ndarray):

    E, C = eigh(H)
    return E[0], C[:, 0]


def subspace_energy(H: np.ndarray, keep: np.ndarray) -> float:
    """Lowest eigenvalue of H restricted to the configurations in keep"""
    keep = np.asarray(keep)
    return eigh(H[np.ix_(keep, keep)], eigvals_only=True)[0]


def weight_order(c0: np.ndarray) -> np.ndarray:
    """Configurations sorted by exact weight, a reference that needs the answer"""
    return np.argsort(np.abs(c0) ** 2, kind="stable")[::-1]


def reference_config(H: np.ndarray, tol: float = 1e-9) -> int:
    """Cheapest configuration on the diagonal, ties broken by second-order energy"""

    diag = np.diag(H)
    refs = np.where(diag <= diag.min() + tol)[0]

    # Several configurations can tie on the diagonal, as every singly filled chain
    # does at large U. Rank those by how much their neighbours lower their energy
    e2 = []
    for i in refs:
        gap = diag[i] - diag
        gap[np.abs(gap) < tol] = np.inf
        e2.append(np.sum(H[:, i] ** 2 / gap))
    return int(refs[np.argmin(e2)])


def epstein_nesbet_order(
    H: np.ndarray, seed: list[int], n_keep: int, batch: int = 8
) -> np.ndarray:
    """Grow a configuration space by second-order perturbation scores, as in CIPSI"""
    # See Huron, Malrieu and Rancurel, J. Chem. Phys. 58, 5745 (1973)
    # https://doi.org/10.1063/1.1679199
    assert n_keep <= len(H), "cannot keep more configurations than exist"
    keep = list(seed)
    diag = np.diag(H)

    while len(keep) < n_keep:
        E, C = eigh(H[np.ix_(keep, keep)])
        psi = np.zeros(len(H))
        psi[keep] = C[:, 0]

        # Score outside configurations by |<D|H|psi>|^2 / |E - H_DD|
        coupling = H @ psi
        gap = np.abs(E[0] - diag)
        gap[gap < 1e-12] = 1e-12  # a configuration degenerate with E would divide by 0
        scores = coupling**2 / gap
        scores[keep] = -1.0

        order = np.argsort(scores, kind="stable")[::-1]
        keep.extend(order[scores[order] >= 0][:batch].tolist())

    return np.array(keep[:n_keep])


def sample_order(c0: np.ndarray, n_shots: int, rng: np.random.Generator):
    """Measure the state n_shots times, return what came back, most frequent first"""
    # qiskit-addon-sqd takes measured configurations and counts like these as input
    # https://github.com/Qiskit/qiskit-addon-sqd/blob/main/qiskit_addon_sqd/subsampling.py
    weights = np.abs(c0) ** 2 / np.sum(np.abs(c0) ** 2)  # circuit states are complex
    draws = rng.choice(len(weights), size=n_shots, p=weights)
    found, counts = np.unique(draws, return_counts=True)
    order = np.argsort(counts, kind="stable")[::-1]
    return found[order], counts[order]


def discovery_curve(
    c0: np.ndarray, shots: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Distinct configurations seen after each number of shots, from one long run"""
    weights = np.abs(c0) ** 2 / np.sum(np.abs(c0) ** 2)  # circuit states are complex
    draws = rng.choice(len(weights), size=int(np.max(shots)), p=weights)

    # A configuration counts as seen at the first shot that drew it
    _, first = np.unique(draws, return_index=True)
    return np.searchsorted(np.sort(first), shots)


def errors_by_size(
    H: np.ndarray, order: np.ndarray, E0: float, sizes: np.ndarray
) -> np.ndarray:
    """Subspace energy error for each prefix of an ordering, nan where it runs out"""
    return np.array(
        [
            subspace_energy(H, order[:k]) - E0 if k <= len(order) else np.nan
            for k in sizes
        ]
    )


def fundamental_rows(labels: np.ndarray) -> np.ndarray:
    """Row of the all-zero state, then the row with one quantum in each mode"""
    total = labels.sum(1)
    ones = [
        np.where((total == 1) & (labels[:, i] == 1))[0][0]
        for i in range(labels.shape[1])
    ]
    return np.array([np.where(total == 0)[0][0]] + ones)


def target_states(C: np.ndarray, rows) -> np.ndarray:
    """Eigenstate that looks most like each row, with the ground state for the first"""
    return np.array([0] + [int(np.argmax(C[row, 1:] ** 2)) + 1 for row in rows[1:]])


def subspace_fundamentals(H: np.ndarray, keep: np.ndarray, rows) -> np.ndarray:
    """Fundamentals from H restricted to keep, nan for modes whose row was not kept"""

    keep = np.asarray(keep)
    nu = np.full(len(rows) - 1, np.nan)
    if len(keep) < 2:
        return nu
    E, C = eigh(H[np.ix_(keep, keep)])
    where = {int(k): n for n, k in enumerate(keep)}
    for i, row in enumerate(rows[1:]):
        if int(row) in where:
            nu[i] = E[target_states(C, [0, where[int(row)]])[1]] - E[0]
    return nu


def fundamental_errors_by_size(
    H: np.ndarray, order: np.ndarray, rows, nu: np.ndarray, sizes: np.ndarray
) -> np.ndarray:
    """Largest fundamental error for each prefix of an ordering, nan until every mode is in"""
    return np.array(
        [
            (
                np.max(np.abs(subspace_fundamentals(H, order[:k], rows) - nu))
                if k <= len(order)
                else np.nan
            )
            for k in sizes
        ]
    )


def epstein_nesbet_targets(
    H: np.ndarray, seeds: list[int], n_keep: int, batch: int = 8
) -> np.ndarray:
    """Grow one space for several states at once, each configuration scored by its best target"""
    # The multi-state version of epstein_nesbet_order. Each seed picks out one target,
    # the eigenstate that looks most like it, and seeds stay at the front of keep
    assert n_keep <= len(H), "cannot keep more configurations than exist"
    keep = list(seeds)
    diag = np.diag(H)

    while len(keep) < n_keep:
        E, C = eigh(H[np.ix_(keep, keep)])
        scores = np.zeros(len(H))
        for state in target_states(C, range(len(seeds))):
            psi = np.zeros(len(H))
            psi[keep] = C[:, state]
            gap = np.abs(E[state] - diag)
            gap[gap < 1e-12] = 1e-12
            scores = np.maximum(scores, (H @ psi) ** 2 / gap)
        scores[keep] = -1.0

        order = np.argsort(scores, kind="stable")[::-1]
        keep.extend(order[scores[order] >= 0][:batch].tolist())

    return np.array(keep[:n_keep])


def sample_targets_order(states: np.ndarray, n_shots: int, rng: np.random.Generator):
    l = len(states.T)
    shotnum = int(n_shots/l)  # shots per state
    totaldraws = []  # collects the results for each state
    for i in range(l):
      # does the shots for each state
      weights = np.abs(states[:,i]) ** 2 / np.sum(np.abs(states[:,i]) ** 2)  # circuit states are complex
      draws = rng.choice(len(weights), size=shotnum, p=weights)
      totaldraws.append(draws)
    alldraws = np.concatenate(totaldraws)  # combines the results into a single array

    # organizes the results together
    found, counts = np.unique(alldraws, return_counts=True)
    order = np.argsort(counts, kind="stable")[::-1]
    return found[order], counts[order]
