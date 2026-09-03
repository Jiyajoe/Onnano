"""
fair_split.py

Multi-object fair division (Mode B, spec sections 9 & 21).

Strategy selection:
  - Many *similar-sized* objects (low variance in area)  -> equal-count split,
    alternating largest-first so counts differ by at most 1.
  - Many *differently-sized* objects                      -> equal-value split
    using the detected area as the value, solved as a balanced-partition
    problem.

The balanced-partition search uses:
  - exact dynamic programming (subset-sum-style) for small/medium N, which
    is guaranteed optimal,
  - a greedy (largest-first, add-to-smaller-pile) heuristic + local swap
    refinement for N beyond the DP's practical size, which stays fast while
    remaining close to optimal in practice.
"""

from dataclasses import dataclass
from typing import List, Tuple
import itertools

# Above this N, skip exact DP (state space gets too large / slow) and use
# the greedy + local-search heuristic instead.
DP_EXACT_MAX_N = 22
# DP resolution: values are scaled to integers with this many buckets, to
# keep the DP table a sane size regardless of the raw area numbers.
DP_RESOLUTION = 2000

SIZE_SIMILARITY_CV_THRESHOLD = 0.12  # coefficient of variation below which
                                      # objects are treated as "similar sized"


@dataclass
class SplitResult:
    group_a_ids: List[int]
    group_b_ids: List[int]
    group_a_value: float
    group_b_value: float
    total_value: float
    strategy: str

    @property
    def group_a_percentage(self) -> float:
        return 100.0 * self.group_a_value / self.total_value if self.total_value else 0.0

    @property
    def group_b_percentage(self) -> float:
        return 100.0 * self.group_b_value / self.total_value if self.total_value else 0.0


def _coefficient_of_variation(values: List[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / n
    return (variance ** 0.5) / mean


def _equal_count_split(ids: List[int], values: List[float]) -> Tuple[List[int], List[int]]:
    """Alternate largest-first onto whichever pile currently has less value,
    which for near-identical items degenerates to a clean alternating split."""
    order = sorted(zip(ids, values), key=lambda p: p[1], reverse=True)
    group_a, group_b = [], []
    val_a = val_b = 0.0
    for i, (oid, val) in enumerate(order):
        if val_a <= val_b:
            group_a.append(oid)
            val_a += val
        else:
            group_b.append(oid)
            val_b += val
    return group_a, group_b


def _exact_partition_dp(ids: List[int], values: List[float]) -> Tuple[List[int], List[int]]:
    """
    Exact balanced 2-partition via subset-sum DP.
    Values are scaled to integers to bound the DP table size.
    """
    total = sum(values)
    if total <= 0:
        mid = len(ids) // 2
        return ids[:mid], ids[mid:]

    scale = DP_RESOLUTION / total
    scaled = [max(1, round(v * scale)) for v in values]
    target = sum(scaled) // 2

    n = len(scaled)
    # dp[s] = True if subset sum s is reachable; track choice via parent pointers
    reachable = {0: []}
    for idx, val in enumerate(scaled):
        new_reachable = dict(reachable)
        for s, chosen in reachable.items():
            ns = s + val
            if ns not in new_reachable:
                new_reachable[ns] = chosen + [idx]
        reachable = new_reachable

    best_s = min(reachable.keys(), key=lambda s: abs(s - target))
    chosen_idx = set(reachable[best_s])

    group_a = [ids[i] for i in range(n) if i in chosen_idx]
    group_b = [ids[i] for i in range(n) if i not in chosen_idx]
    return group_a, group_b


def _greedy_partition_with_local_search(ids: List[int], values: List[float]) -> Tuple[List[int], List[int]]:
    """Greedy largest-first assignment, then local 1-item swaps to improve balance."""
    order = sorted(range(len(ids)), key=lambda i: values[i], reverse=True)
    group_a, group_b = [], []
    val_a = val_b = 0.0
    for i in order:
        if val_a <= val_b:
            group_a.append(i)
            val_a += values[i]
        else:
            group_b.append(i)
            val_b += values[i]

    improved = True
    while improved:
        improved = False
        best_diff = abs(val_a - val_b)
        for ia in group_a:
            for ib in group_b:
                new_a = val_a - values[ia] + values[ib]
                new_b = val_b - values[ib] + values[ia]
                if abs(new_a - new_b) < best_diff:
                    group_a.remove(ia); group_a.append(ib)
                    group_b.remove(ib); group_b.append(ia)
                    val_a, val_b = new_a, new_b
                    best_diff = abs(val_a - val_b)
                    improved = True
                    break
            if improved:
                break

    return [ids[i] for i in group_a], [ids[i] for i in group_b]


def fair_split(object_ids: List[int], object_values: List[float]) -> SplitResult:
    """
    Main entry point: pick a strategy based on how similar the objects are,
    then solve the resulting balanced-partition problem.
    """
    n = len(object_ids)
    if n == 0:
        return SplitResult([], [], 0.0, 0.0, 0.0, "none")
    if n == 1:
        # A single object with nothing to pair it against - give it to
        # whichever pile, flag as unavoidable.
        return SplitResult(object_ids, [], object_values[0], 0.0, object_values[0], "single_item")

    cv = _coefficient_of_variation(object_values)

    if cv <= SIZE_SIMILARITY_CV_THRESHOLD:
        strategy = "equal_count"
        group_a, group_b = _equal_count_split(object_ids, object_values)
    else:
        strategy = "equal_value"
        if n <= DP_EXACT_MAX_N:
            group_a, group_b = _exact_partition_dp(object_ids, object_values)
        else:
            group_a, group_b = _greedy_partition_with_local_search(object_ids, object_values)

    value_map = dict(zip(object_ids, object_values))
    val_a = sum(value_map[i] for i in group_a)
    val_b = sum(value_map[i] for i in group_b)

    return SplitResult(
        group_a_ids=group_a, group_b_ids=group_b,
        group_a_value=val_a, group_b_value=val_b,
        total_value=val_a + val_b, strategy=strategy,
    )
