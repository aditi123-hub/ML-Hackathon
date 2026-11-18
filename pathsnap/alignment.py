"""Assign GPS fixes in order along an already specified directed arc sequence."""

from dataclasses import dataclass

import numpy as np

from .geometry import RoadGeometry


@dataclass
class AlignedReference:
    """One arc occurrence and its spatial residual for each aligned observation."""

    arcs: np.ndarray
    route_positions: np.ndarray
    distance_m: np.ndarray


def align_reference(road: RoadGeometry, reference: list[int]) -> AlignedReference:
    """Associate GPS observations to the untimed development reference."""
    return align_ordered_arcs(road, reference)


def align_ordered_arcs(
    road: RoadGeometry, route: list[int], rows: slice = slice(None), candidate_ids: np.ndarray | None = None,
) -> AlignedReference:
    """Find the minimum-cost monotone assignment to route arc occurrences.

    The per-observation cost combines segment distance and heading agreement.
    A linear forward-distance penalty discourages jumps. Prefix minima reduce
    the recurrence from O(observations * route_arcs²) to O(observations *
    route_arcs). Backpointers keep repeated occurrences of one arc distinct.

    The caller supplies the sequence. The decoder passes its own predicted
    block. Reference scoring passes the stored route separately after prediction.
    """
    # Route positions are independent of graph IDs. A loop may require the
    # same arc ID to appear in several distinct positions.
    ref = np.asarray(route, dtype=np.int32)
    positions = np.arange(len(ref))
    points, headings, gaps = road.points[rows], road.headings[rows], road.prior_gap[rows]
    lengths = road.lengths[ref]
    # Midpoint offsets provide one increasing distance coordinate along the route.
    along_route = np.cumsum(lengths) - 0.5 * lengths
    directions = np.divide(
        road.vectors[ref], lengths[:, None], out=np.zeros((len(ref), 2)),
        where=lengths[:, None] > 0,
    )
    back = np.zeros(
        (len(points), len(ref)),
        dtype=np.uint16 if len(ref) <= 65536 else np.uint32,
    )
    previous = np.zeros(len(ref))
    available = np.zeros(len(road.arcs), dtype=bool) if candidate_ids is not None else None
    for row, (point, heading, gap) in enumerate(zip(points, headings, gaps)):
        distances, _ = road.distance_and_fraction(point, ref)
        # Distances are in metres. Opposite heading contributes up to 12 m.
        emission = distances + 6 * (1 - np.sum(directions * heading, axis=1))
        if available is not None:
            # By restricting the final reassignment to each fix's candidates,
            # the decoder cannot give a fix an arc it could not retrieve.
            candidates = candidate_ids[row]
            valid_ids = candidates[candidates >= 0]
            available[valid_ids] = True
            emission[~available[ref]] = np.inf
            available[valid_ids] = False
        if row == 0:
            # The GPS track can start mid-route. Allow all starting positions.
            previous = emission
            continue
        # Long gaps weaken the displacement prior while still preserving order.
        weight = 0.025 if 0 < gap <= 20 else 0.004
        offset = weight * along_route
        # The cost from earlier position k to later j is w*(s[j] - s[k]).
        # For a fixed j, subtract w*s[k] before taking a prefix minimum over
        # k <= j, then add w*s[j] once. This avoids a quadratic edge search.
        adjusted = previous - offset
        minimum = np.minimum.accumulate(adjusted)
        # Track where the prefix minimum was reached to recover the path.
        back[row] = np.maximum.accumulate(
            np.where(np.isfinite(adjusted) & (adjusted == minimum), positions, 0)
        )
        previous = emission + offset + minimum

    if not np.isfinite(previous).any():
        raise ValueError("No monotone assignment to the supplied route")
    path = np.empty(len(points), dtype=np.int32)
    # End anywhere on the route and follow the saved positions backwards.
    path[-1] = np.argmin(previous)
    for row in range(len(path) - 1, 0, -1):
        path[row - 1] = back[row, path[row]]
    matched = ref[path]
    distance, _ = road.distance_and_fraction(points, matched)
    return AlignedReference(matched, path, distance)
