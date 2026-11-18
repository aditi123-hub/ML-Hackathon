"""Audit the PathSnap student archive and plot one selected development trip."""

import argparse
import json
from itertools import pairwise
from math import isfinite
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from .dataset import Trip, load_reference_route, load_trip, trip_member
from .inventory import EXPECTED_TRIPS, FILE_SUFFIXES, inventory


def _root(parents: list[int], node: int) -> int:
    while parents[node] >= 0:
        node = parents[node]
    return node


def network_stats(trip: Trip) -> dict[str, int]:
    """Count weak components, including singleton nodes, using union by size."""
    parents = [-1] * len(trip.nodes)
    incident = bytearray(len(trip.nodes))
    components = len(trip.nodes)
    for source, target in trip.arcs:
        incident[source] = incident[target] = 1
        left = _root(parents, source)
        right = _root(parents, target)
        if left == right:
            continue
        if parents[left] > parents[right]:
            left, right = right, left
        parents[left] += parents[right]
        parents[right] = left
        components -= 1

    return {
        "weak_components": components,
        "largest_component_nodes": -min(parents),
        "isolated_nodes": incident.count(0),
    }


def audit_trip(archive: ZipFile, trip: Trip, route: list[int]) -> dict:
    longitudes, latitudes, timestamps = zip(*trip.observations)
    intervals = [later - earlier for earlier, later in pairwise(timestamps)]
    if not all(isfinite(interval) for interval in intervals):
        raise ValueError(f"{trip.split}/{trip.trip_id}: timestamp interval exceeds finite range")
    return {
        "trip_id": trip.trip_id,
        "file_bytes": {
            suffix: archive.getinfo(trip_member(trip.split, trip.trip_id, suffix)).file_size
            for suffix in sorted(FILE_SUFFIXES)
        },
        "gps_observations": len(trip.observations),
        "bounds": {
            "min_longitude": min(longitudes),
            "max_longitude": max(longitudes),
            "min_latitude": min(latitudes),
            "max_latitude": max(latitudes),
        },
        "timestamp_steps": len(intervals),
        "subsecond_steps": sum(0 < interval < 1 for interval in intervals),
        "one_second_steps": intervals.count(1),
        "gaps_over_one_second": sum(interval > 1 for interval in intervals),
        "largest_gap_seconds": max(0.0, max(intervals, default=0.0)),
        "duplicate_timestamp_steps": intervals.count(0),
        "backward_timestamp_steps": sum(interval < 0 for interval in intervals),
        "nodes": len(trip.nodes),
        "directed_arcs": len(trip.arcs),
        **network_stats(trip),
        "reference_route_arcs": len(route),
        "reference_discontinuities": sum(
            trip.arcs[previous][1] != trip.arcs[current][0]
            for previous, current in pairwise(route)
        ),
    }


def summarize_trips(stats: list[dict]) -> dict:
    largest_gap = max(stats, key=lambda trip: trip["largest_gap_seconds"])
    return {
        "trips": len(stats),
        "file_count": sum(len(trip["file_bytes"]) for trip in stats),
        "gps_observations": sum(trip["gps_observations"] for trip in stats),
        "timestamp_steps": sum(trip["timestamp_steps"] for trip in stats),
        "subsecond_steps": sum(trip["subsecond_steps"] for trip in stats),
        "one_second_steps": sum(trip["one_second_steps"] for trip in stats),
        "gaps_over_one_second": sum(trip["gaps_over_one_second"] for trip in stats),
        "largest_gap_seconds": largest_gap["largest_gap_seconds"],
        "largest_gap_trip": largest_gap["trip_id"],
        "duplicate_timestamp_steps": sum(trip["duplicate_timestamp_steps"] for trip in stats),
        "backward_timestamp_steps": sum(trip["backward_timestamp_steps"] for trip in stats),
        "nodes": sum(trip["nodes"] for trip in stats),
        "directed_arcs": sum(trip["directed_arcs"] for trip in stats),
        "isolated_nodes": sum(trip["isolated_nodes"] for trip in stats),
        "trips_with_multiple_components": sum(trip["weak_components"] > 1 for trip in stats),
        "reference_route_arcs": sum(trip["reference_route_arcs"] for trip in stats),
        "reference_discontinuities": sum(trip["reference_discontinuities"] for trip in stats),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to Student folder.zip")
    parser.add_argument("--output", type=Path, default=Path("outputs/eda"))
    parser.add_argument("--map-split", choices=EXPECTED_TRIPS, default="train")
    parser.add_argument("--map-trip", default="00000014", help="Trip ID to visualize")
    args = parser.parse_args()

    try:
        trips = inventory(args.archive)
        if args.map_trip not in trips[args.map_split]:
            raise ValueError(f"Trip {args.map_trip} is not in {args.map_split}")

        from .visualization import write_trajectory_figure, write_trip_map

        report = {}
        with ZipFile(args.archive) as archive:
            for split, split_trips in trips.items():
                stats = []
                for trip_id in sorted(split_trips):
                    trip = load_trip(archive, split, trip_id)
                    route = load_reference_route(archive, trip)
                    stats.append(audit_trip(archive, trip, route))
                report[split] = {"summary": summarize_trips(stats), "trips": stats}

            example = load_trip(archive, args.map_split, args.map_trip)
            reference = load_reference_route(archive, example)

        args.output.mkdir(parents=True, exist_ok=True)
        stem = f"{example.split}-{example.trip_id}"
        write_trajectory_figure(example, reference, args.output / f"{stem}.png")
        write_trip_map(example, reference, args.output / f"{stem}.html")
        (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (BadZipFile, OSError, ValueError) as error:
        parser.exit(1, f"Dataset exploration failed: {error}\n")

    for split, partition in report.items():
        summary = partition["summary"]
        print(
            f"{split}: {summary['trips']} trips, "
            f"{summary['gps_observations']:,} GPS observations, "
            f"{summary['reference_discontinuities']} reference discontinuities"
        )
    print(f"Audit, plot and map saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
