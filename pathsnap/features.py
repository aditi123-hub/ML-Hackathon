"""Generate candidate road arcs from GPS and road geometry alone."""

from dataclasses import dataclass

import numpy as np
import shapely
from shapely import STRtree

from .geometry import RoadGeometry
from .trajectory import CONTEXT_WINDOWS_S, TrajectoryContext


# This order is part of the model artifact contract. The first checkpoint
# used these 12 features. Context appends columns and preserves their indices.
LEGACY_FEATURE_NAMES = (
    "distance_m", "distance_above_nearest_m", "arc_length_m",
    "heading_cosine", "heading_available", "arc_fraction",
    "previous_distance_m", "previous_available", "next_distance_m", "next_available",
    "previous_gap_s", "next_gap_s",
)
FEATURE_NAMES = LEGACY_FEATURE_NAMES + (
    "smoothed_distance_m", "smoothed_distance_above_nearest_m",
    "incoming_5s_distance_m", "outgoing_5s_distance_m",
) + tuple(
    f"{side}_{seconds}s_{measure}"
    for side in ("incoming", "outgoing")
    for seconds in CONTEXT_WINDOWS_S
    for measure in ("heading_cosine", "speed_mps")
)


@dataclass(frozen=True)
class CandidateConfig:
    """Shared training/inference settings, saved alongside the fitted model."""

    radius_m: float = 150.0
    limit: int = 20
    heading_gap_s: float = 20.0
    heading_displacement_m: float = 2.0
    trajectory_context: bool = True

    def __post_init__(self):
        if not all(np.isfinite(x) and x > 0 for x in (
            self.radius_m, self.limit, self.heading_gap_s, self.heading_displacement_m,
        )):
            raise ValueError("Candidate parameters must be finite and positive")

    @property
    def feature_names(self):
        return FEATURE_NAMES if self.trajectory_context else LEGACY_FEATURE_NAMES

    @classmethod
    def from_schema(cls, settings, feature_names):
        # Original artifacts and caches omit a feature-version switch.
        config = cls(**({"trajectory_context": False} | settings))
        if config.feature_names != tuple(feature_names):
            raise ValueError("Feature schema does not match the candidate configuration")
        return config


@dataclass
class Candidates:
    """Fixed-width candidate slots per fix. An arc ID of -1 denotes padding."""

    arc_ids: np.ndarray
    features: np.ndarray

    @property
    def valid(self):
        return self.arc_ids >= 0

    @property
    def fractions(self):
        return self.features[:, :, FEATURE_NAMES.index("arc_fraction")]


def generate_candidates(road: RoadGeometry, config: CandidateConfig) -> Candidates:
    """Find nearby finite segments with an R-tree, then rank by exact distance."""
    rows, arcs = query_candidate_pairs(road, config.radius_m)
    observation_count = len(road.points)
    distances, fractions = road.distance_and_fraction(road.points[rows], arcs)
    # Group the flat pairs by observation, then sort by distance and arc ID.
    # The arc ID tie-breaker matters on two-way roads with identical geometry.
    order = np.lexsort((arcs, distances, rows))
    rows, arcs = rows[order], arcs[order]
    distances, fractions = distances[order], fractions[order]
    # The first offset indexes the closest arc. Subtract each observation's
    # offset from absolute row positions to get zero-based candidate ranks.
    counts = np.bincount(rows, minlength=observation_count)
    offsets = np.r_[0, np.cumsum(counts)[:-1]]
    ranks = np.arange(len(rows)) - offsets[rows]
    keep = ranks < config.limit
    rows, arcs, ranks = rows[keep], arcs[keep], ranks[keep]
    nearest_distances = distances[offsets]
    distances, fractions = distances[keep], fractions[keep]
    # Keep fixed-width slots for model batches and Viterbi. Only actual
    # candidates are populated or scored, so padding is never a road arc.
    ids = np.full((observation_count, config.limit), -1, dtype=np.int32)
    ids[rows, ranks] = arcs

    # A dot product between unit directions is +1 along travel and -1 against it.
    heading_ok = road.has_heading[rows] & (road.lengths[arcs] > 0)
    heading = np.zeros(len(rows))
    heading[heading_ok] = (
        np.sum(road.headings[rows[heading_ok]] * road.vectors[arcs[heading_ok]], axis=1)
        / road.lengths[arcs[heading_ok]]
    )

    previous, previous_known = neighbor_distances(road, rows, arcs, -1, config.heading_gap_s)
    following, following_known = neighbor_distances(road, rows, arcs, 1, config.heading_gap_s)
    features = np.zeros((*ids.shape, len(config.feature_names)), dtype=np.float32)
    # Keep the exact feature schema order above when assembling model inputs.
    local_features = (
        distances, distances - nearest_distances[rows], road.lengths[arcs],
        heading, heading_ok, fractions, previous, previous_known, following,
        following_known, np.clip(road.prior_gap[rows], 0, config.heading_gap_s),
        np.clip(road.next_gap[rows], 0, config.heading_gap_s),
    )
    if config.trajectory_context:
        context = TrajectoryContext.build(road.points, road.timestamps, config.heading_gap_s)
        local_features += trajectory_features(road, context, rows, arcs)
    features[rows, ranks] = np.column_stack(local_features)
    return Candidates(ids, features)


def query_candidate_pairs(road: RoadGeometry, radius_m: float):
    """Return observation and arc row indices for the radius hits and fallbacks."""
    # The index sees the same metre coordinate plane as the numeric distance code.
    lines = shapely.linestrings(road.nodes[road.arcs])
    tree = STRtree(lines)
    points = shapely.points(road.points)
    rows, arcs = tree.query(points, predicate="dwithin", distance=radius_m)
    empty = np.flatnonzero(np.bincount(rows, minlength=len(points)) == 0)
    if empty.size:
        # Find the nearest arc only for fixes with no radius hit. Keep distance
        # ties so the shared sorting step gives the same result across runs.
        nearest_rows, nearest_arcs = tree.query_nearest(points[empty], all_matches=True)
        rows = np.r_[rows, empty[nearest_rows]]
        arcs = np.r_[arcs, nearest_arcs]
    return rows, arcs


def neighbor_distances(road, rows, arcs, step, max_gap_s):
    """Measure an adjacent fix against the same arc when the gap is valid."""
    gaps = road.prior_gap if step == -1 else road.next_gap
    known = (gaps[rows] > 0) & (gaps[rows] <= max_gap_s)
    # Unknown entries stay zero with a false mask. This distinguishes a missing
    # neighbor from a measured distance of zero for a point on the road.
    values = np.zeros(len(rows))
    values[known], _ = road.distance_and_fraction(road.points[rows[known] + step], arcs[known])
    return values, known


def trajectory_features(road, context, rows, arcs):
    """Measure spatial fit and travel direction over several time scales."""
    smoothed, _ = road.distance_and_fraction(context.points[rows], arcs)
    best = np.full(len(road.points), np.inf)
    np.minimum.at(best, rows, smoothed)  # Reduce competing arcs for each observation.
    before, _ = road.distance_and_fraction(context.points[context.previous[0, rows]], arcs)
    after, _ = road.distance_and_fraction(context.points[context.following[0, rows]], arcs)
    columns = [smoothed, smoothed - best[rows], before, after]

    lengths = road.lengths[arcs]
    directions = np.divide(
        road.vectors[arcs], lengths[:, None], where=lengths[:, None] > 0,
        out=np.zeros((len(arcs), 2)),
    )
    # The short window helps at junctions. Wider windows can disambiguate arc
    # direction where adjacent readings mostly reflect GPS drift.
    for offset, interval in zip(context.offsets, context.elapsed):
        displacement = np.linalg.norm(offset, axis=1)
        speed = np.divide(displacement, interval, out=np.zeros_like(interval), where=interval > 0)
        cosine = np.divide(
            np.einsum("ij,ij->i", directions, offset[rows]), displacement[rows],
            out=np.zeros(len(rows)), where=displacement[rows] >= 2,
        )
        columns.extend((cosine, speed[rows]))
    return tuple(columns)
