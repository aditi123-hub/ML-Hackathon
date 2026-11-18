"""Generate candidate road arcs from GPS and road geometry alone."""

from dataclasses import dataclass

import numpy as np
import shapely
from pyproj import Proj
from shapely import STRtree

from .dataset import Trip


FEATURE_NAMES = (
    "distance_m", "distance_above_nearest_m", "arc_length_m",
    "heading_cosine", "heading_available", "arc_fraction",
    "previous_distance_m", "previous_available", "next_distance_m", "next_available",
    "previous_gap_s", "next_gap_s",
)


@dataclass(frozen=True)
class CandidateConfig:
    radius_m: float = 100.0
    limit: int = 16
    heading_gap_s: float = 20.0
    heading_displacement_m: float = 2.0

    def __post_init__(self):
        if not self.radius_m > 0 or self.limit < 1:
            raise ValueError("Candidate radius and limit must be positive")


@dataclass
class RoadGeometry:
    nodes: np.ndarray
    arcs: np.ndarray
    starts: np.ndarray
    vectors: np.ndarray
    lengths: np.ndarray
    points: np.ndarray
    timestamps: np.ndarray
    headings: np.ndarray
    has_heading: np.ndarray
    prior_gap: np.ndarray
    next_gap: np.ndarray

    @classmethod
    def from_trip(cls, trip: Trip, config: CandidateConfig):
        track = np.asarray(trip.observations, dtype=np.float64)
        # A trip-local azimuthal equidistant projection provides coordinates in
        # metres and avoids longitude wrap at the antimeridian.
        lon, lat = track[0, :2]
        projection = Proj(proj="aeqd", lon_0=lon, lat_0=lat, ellps="WGS84")

        def project(coordinates):
            x, y = projection(coordinates[:, 0], coordinates[:, 1], errcheck=True)
            return np.column_stack((x, y))

        nodes = project(np.asarray(trip.nodes))
        points = project(track[:, :2])
        arcs = np.asarray(trip.arcs, dtype=np.int32)
        starts = nodes[arcs[:, 0]]
        vectors = nodes[arcs[:, 1]] - starts
        lengths = np.linalg.norm(vectors, axis=1)
        times = track[:, 2]

        delta_t = np.diff(times)
        steps = np.diff(points, axis=0)
        step_lengths = np.linalg.norm(steps, axis=1)
        usable = ((delta_t > 0) & (delta_t <= config.heading_gap_s)
                  & (step_lengths >= config.heading_displacement_m))
        directions = np.zeros_like(steps)
        directions[usable] = steps[usable] / step_lengths[usable, None]
        # Average the reliable incoming and outgoing unit vectors. Keep a mask
        # so a stop, dropout, or cancellation does not look like a heading.
        headings = np.zeros_like(points)
        headings[:-1] += directions
        headings[1:] += directions
        magnitude = np.linalg.norm(headings, axis=1)
        has_heading = magnitude > 0.1
        headings[has_heading] /= magnitude[has_heading, None]
        return cls(
            nodes, arcs, starts, vectors, lengths, points, times, headings,
            has_heading, np.r_[0, delta_t], np.r_[delta_t, 0],
        )

    def distance_and_fraction(self, points, arc_ids):
        """Distance to the finite segment and position along its source-to-target axis."""
        vectors = self.vectors[arc_ids]
        delta = points - self.starts[arc_ids]
        length_squared = self.lengths[arc_ids] ** 2
        dot = np.sum(delta * vectors, axis=-1)
        fraction = np.divide(dot, length_squared, out=np.zeros_like(dot), where=length_squared > 0)
        np.clip(fraction, 0, 1, out=fraction)
        return np.linalg.norm(delta - fraction[..., None] * vectors, axis=-1), fraction


@dataclass
class Candidates:
    # Padding has arc_id = -1. Road IDs stay local to each trip.
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
    lines = shapely.linestrings(road.nodes[road.arcs])
    tree = STRtree(lines)
    points = shapely.points(road.points)
    rows, arcs = tree.query(points, predicate="dwithin", distance=config.radius_m)
    empty = np.flatnonzero(np.bincount(rows, minlength=len(points)) == 0)
    if empty.size:
        # A nearest segment keeps inference defined for an outlier outside the
        # search radius. Keep ties so the final ID-based ordering is deterministic.
        nearest_rows, nearest_arcs = tree.query_nearest(points[empty], all_matches=True)
        rows = np.r_[rows, empty[nearest_rows]]
        arcs = np.r_[arcs, nearest_arcs]

    distances, fractions = road.distance_and_fraction(road.points[rows], arcs)
    order = np.lexsort((arcs, distances, rows))
    rows, arcs = rows[order], arcs[order]
    distances, fractions = distances[order], fractions[order]
    counts = np.bincount(rows, minlength=len(points))
    offsets = np.r_[0, np.cumsum(counts)[:-1]]
    ranks = np.arange(len(rows)) - offsets[rows]
    keep = ranks < config.limit
    rows, arcs, ranks = rows[keep], arcs[keep], ranks[keep]
    nearest_distances = distances[offsets]
    distances, fractions = distances[keep], fractions[keep]
    ids = np.full((len(points), config.limit), -1, dtype=np.int32)
    ids[rows, ranks] = arcs

    heading_ok = road.has_heading[rows] & (road.lengths[arcs] > 0)
    heading = np.zeros(len(rows))
    heading[heading_ok] = (
        np.sum(road.headings[rows[heading_ok]] * road.vectors[arcs[heading_ok]], axis=1)
        / road.lengths[arcs[heading_ok]]
    )

    def neighbor_features(neighbor, gap):
        known = (gap[rows] > 0) & (gap[rows] <= config.heading_gap_s)
        values = np.zeros(len(rows))
        values[known], _ = road.distance_and_fraction(road.points[neighbor[known]], arcs[known])
        return values, known

    previous, previous_known = neighbor_features(rows - 1, road.prior_gap)
    following, following_known = neighbor_features(rows + 1, road.next_gap)
    features = np.zeros((*ids.shape, len(FEATURE_NAMES)), dtype=np.float32)
    features[rows, ranks] = np.column_stack((
        distances, distances - nearest_distances[rows], road.lengths[arcs],
        heading, heading_ok, fractions, previous, previous_known, following,
        following_known, np.clip(road.prior_gap[rows], 0, config.heading_gap_s),
        np.clip(road.next_gap[rows], 0, config.heading_gap_s),
    ))
    return Candidates(ids, features)
