
"""Generate nearby directed road-arc candidates for GPS observations."""

from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely import STRtree
from shapely.geometry import LineString, Point


def load_track(path):
    return np.atleast_2d(np.loadtxt(path))


def load_nodes(path):
    return np.atleast_2d(np.loadtxt(path))


def load_arcs(path):
    return np.atleast_2d(np.loadtxt(path, dtype=int))


def project_coordinates(coordinates):
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32632", always_xy=True
    )
    x, y = transformer.transform(
        coordinates[:, 0], coordinates[:, 1]
    )
    return np.column_stack((x, y))


def generate_candidates(track, nodes, arcs, radius_m=50):
    """Return nearby arcs with their IDs and point-to-arc distances."""

    if radius_m < 0:
        raise ValueError("radius_m must be non-negative")

    if len(arcs) and (
        np.any(arcs < 0) or np.any(arcs >= len(nodes))
    ):
        raise ValueError("Arc contains an invalid node ID")

    gps_xy = project_coordinates(track[:, :2])
    nodes_xy = project_coordinates(nodes)

    road_segments = [
        LineString([nodes_xy[start], nodes_xy[end]])
        for start, end in arcs
    ]

    if not road_segments:
        return [[] for _ in range(len(track))]

    tree = STRtree(road_segments)
    results = []

    for observation_index, coordinates in enumerate(gps_xy):
        point = Point(coordinates)
        nearby_indices = tree.query(
            point, predicate="dwithin", distance=radius_m
        )

        candidates = []

        for arc_id in nearby_indices:
            distance = point.distance(road_segments[int(arc_id)])

            candidates.append({
                "observation_index": observation_index,
                "arc_id": int(arc_id),
                "distance_m": float(distance),
            })

        candidates.sort(key=lambda item: item["distance_m"])
        results.append(candidates)

    return results


if __name__ == "__main__":
    base = (
        Path.home() / "Downloads" / "PathSnap-data"
        / "Student folder" / "train" / "00000001"
    )

    track = load_track(base / "00000001.track")
    nodes = load_nodes(base / "00000001.nodes")
    arcs = load_arcs(base / "00000001.arcs")

    candidates = generate_candidates(track, nodes, arcs)

    total = sum(len(items) for items in candidates)

    print("GPS observations:", len(track))
    print("Road nodes:", len(nodes))
    print("Road arcs:", len(arcs))
    print("Observations with candidates:",
          sum(bool(items) for items in candidates))
    print("Total candidates:", total)

    if total:
        distances = [
            item["distance_m"]
            for items in candidates
            for item in items
        ]
        print("Maximum candidate distance (m):", max(distances))
        print("All candidate IDs valid:",
              all(0 <= item["arc_id"] < len(arcs)
                  for items in candidates for item in items))
        print("All distances within radius:",
              all(distance <= 50 for distance in distances))
