"""Infer observation labels from the ordered, untimed development reference."""

from dataclasses import dataclass

import numpy as np

from .features import RoadGeometry


@dataclass
class AlignedReference:
    arcs: np.ndarray
    route_positions: np.ndarray
    distance_m: np.ndarray


def align_reference(road: RoadGeometry, reference: list[int]) -> AlignedReference:
    """Find the minimum-cost monotone assignment to reference arc occurrences.

    The per-observation cost combines segment distance and heading agreement.
    A linear forward-distance penalty discourages jumps. Prefix minima reduce
    the recurrence from O(observations * reference_arcs²) to O(observations *
    reference_arcs). Backpointers keep repeated occurrences of one arc distinct.
    """
    ref = np.asarray(reference, dtype=np.int32)
    positions = np.arange(len(ref))
    lengths = road.lengths[ref]
    along_route = np.cumsum(lengths) - 0.5 * lengths
    directions = np.divide(
        road.vectors[ref], lengths[:, None], out=np.zeros((len(ref), 2)),
        where=lengths[:, None] > 0,
    )
    back = np.zeros(
        (len(road.points), len(ref)),
        dtype=np.uint16 if len(ref) <= 65536 else np.uint32,
    )
    previous = np.zeros(len(ref))
    for row, point in enumerate(road.points):
        distances, _ = road.distance_and_fraction(point, ref)
        # Distances are in metres. Opposite heading contributes up to 12 m.
        emission = distances + 6 * (1 - np.sum(directions * road.headings[row], axis=1))
        if row == 0:
            previous = emission
            continue
        # Long gaps weaken the displacement prior while still preserving order.
        weight = 0.025 if 0 < road.prior_gap[row] <= 20 else 0.004
        offset = weight * along_route
        adjusted = previous - offset
        minimum = np.minimum.accumulate(adjusted)
        back[row] = np.maximum.accumulate(np.where(adjusted == minimum, positions, 0))
        previous = emission + offset + minimum

    path = np.empty(len(road.points), dtype=np.int32)
    path[-1] = np.argmin(previous)
    for row in range(len(path) - 1, 0, -1):
        path[row - 1] = back[row, path[row]]
    matched = ref[path]
    distance, _ = road.distance_and_fraction(road.points, matched)
    return AlignedReference(matched, path, distance)
