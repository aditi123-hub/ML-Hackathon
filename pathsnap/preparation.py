"""Prepare a reproducible cache for the training and student-test partitions."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .alignment import align_reference
from .dataset import load_reference_route, load_trip
from .features import CandidateConfig, FEATURE_NAMES, RoadGeometry, generate_candidates
from .inventory import inventory, EXPECTED_TRIPS


@dataclass
class PreparedTrip:
    trip_id: str
    candidate_ids: np.ndarray
    features: np.ndarray
    reference_ids: np.ndarray
    reference_distances: np.ndarray
    arc_nodes: np.ndarray
    gaps: np.ndarray


def prepare(archive_path: Path, destination: Path, config: CandidateConfig) -> dict:
    splits = inventory(archive_path)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "features": FEATURE_NAMES,
        "candidate_config": asdict(config),
        "splits": {},
    }
    with ZipFile(archive_path) as archive:
        for split, trip_ids in splits.items():
            folder = destination / split
            folder.mkdir(exist_ok=True)
            manifest["splits"][split] = []
            for trip_id in sorted(trip_ids):
                trip = load_trip(archive, split, trip_id)
                road = RoadGeometry.from_trip(trip, config)
                candidates = generate_candidates(road, config)
                # Label preparation is a separate pass. The candidate set has
                # already been fixed and cannot be amended by the reference.
                label = align_reference(road, load_reference_route(archive, trip))
                matches = candidates.arc_ids == label.arcs[:, None]
                distances = candidates.features[:, :, 0]
                diagnostics = {
                    "trip_id": trip_id,
                    "observations": len(label.arcs),
                    "mean_candidates": float(candidates.valid.sum() / len(label.arcs)),
                    "reference_beyond_100m": int(np.count_nonzero(label.distance_m > 100)),
                    "misses": int(np.count_nonzero(~matches.any(axis=1))),
                    "radius_grid": {},
                }
                # The 100-m set also contains all nearer candidates up to the
                # same cap. Always retain the first (nearest) as the fallback.
                for radius in (40, 70, 100, 150):
                    if radius <= config.radius_m:
                        available = distances <= radius
                        available[:, 0] = True
                        diagnostics["radius_grid"][str(radius)] = int(
                            np.count_nonzero(~(matches & available).any(axis=1))
                        )
                manifest["splits"][split].append(diagnostics)
                np.savez_compressed(
                    folder / f"{trip_id}.npz",
                    candidate_ids=candidates.arc_ids,
                    features=candidates.features,
                    reference_ids=label.arcs,
                    reference_distances=label.distance_m.astype(np.float32),
                    arc_nodes=road.arcs,
                    gaps=road.prior_gap,
                )
            observations = sum(item["observations"] for item in manifest["splits"][split])
            misses = sum(item["misses"] for item in manifest["splits"][split])
            print(f"{split}: {len(trip_ids)} trips, {observations:,} points, candidate recall {1 - misses / observations:.2%}", flush=True)

    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_prepared(root: Path, split: str, manifest: dict) -> list[PreparedTrip]:
    records = manifest["splits"][split]
    ids = [record["trip_id"] for record in records]
    if len(ids) != EXPECTED_TRIPS[split] or len(set(ids)) != len(ids):
        raise ValueError(f"Unexpected trip list for {split}")
    batches = []
    for trip_id in ids:
        with np.load(root / split / f"{trip_id}.npz", allow_pickle=False) as arrays:
            batches.append(PreparedTrip(trip_id, **dict(arrays)))
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/prepared"))
    parser.add_argument("--radius", type=float, default=CandidateConfig.radius_m)
    parser.add_argument("--limit", type=int, default=CandidateConfig.limit)
    args = parser.parse_args()
    prepare(args.archive, args.output, CandidateConfig(radius_m=args.radius, limit=args.limit))


if __name__ == "__main__":
    main()
