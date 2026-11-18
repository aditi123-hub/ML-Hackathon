"""Prepare a reproducible cache for the training and student-test partitions."""

import argparse
import csv
from dataclasses import asdict, dataclass
from hashlib import file_digest, sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from .alignment import align_reference
from .clean_data import FIT_REFERENCE_DISTANCE_M, csv_tables, validate_observation_order, write_trip
from .dataset import load_reference_route, load_trip
from .features import CandidateConfig, generate_candidates
from .geometry import RoadGeometry
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
    fit_eligible: np.ndarray | None = None


def prepare(archive_path: Path, destination: Path, config: CandidateConfig) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    with archive_path.open("rb") as stream:
        archive_hash = file_digest(stream, "sha256").hexdigest()
    manifest = {
        "schema_version": 2,
        "archive_sha256": archive_hash,
        "features": config.feature_names,
        "candidate_config": asdict(config),
        "fit_reference_distance_m": FIT_REFERENCE_DISTANCE_M,
    }
    key = sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    manifest["bundle"] = f"bundles/{key}"
    bundle = destination / manifest["bundle"]
    bundle.parent.mkdir(exist_ok=True)

    # Publish the manifest only after the CSV tables and feature cache finish.
    # Older manifests remain readable during a rebuild or a failed validation.
    if not bundle.exists():
        with TemporaryDirectory(dir=bundle.parent) as staging:
            staged_bundle = Path(staging) / "bundle"
            staged_bundle.mkdir()
            build_bundle(archive_path, staged_bundle, config, manifest)
            try:
                staged_bundle.rename(bundle)
            except FileExistsError:
                # Another run published the same archive and configuration.
                pass
    else:
        print(f"Reusing prepared bundle {bundle}", flush=True)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    with NamedTemporaryFile(dir=destination, delete=False) as stream:
        temporary_manifest = Path(stream.name)
        stream.write((bundle / "manifest.json").read_bytes())
    temporary_manifest.replace(destination / "manifest.json")
    print(f"Clean CSV tables: {bundle / 'clean'}", flush=True)
    return manifest


def build_bundle(archive_path, destination, config, manifest):
    """Write normalized tables and indexed NumPy batches for development only."""
    splits = inventory(archive_path)
    manifest["splits"] = {}
    with ZipFile(archive_path) as archive:
        for split, trip_ids in splits.items():
            folder = destination / split
            folder.mkdir()
            manifest["splits"][split] = []
            with csv_tables(destination / "clean" / split) as tables:
                for index, trip_id in enumerate(sorted(trip_ids), start=1):
                    diagnostics = prepare_trip(archive, split, trip_id, config, tables, folder)
                    manifest["splits"][split].append(diagnostics)
                    if index % 10 == 0:
                        print(f"Prepared {split}: {index}/{len(trip_ids)} trips", flush=True)
            observations = sum(item["observations"] for item in manifest["splits"][split])
            misses = sum(item["misses"] for item in manifest["splits"][split])
            print(f"{split}: {len(trip_ids)} trips, {observations:,} points, candidate recall {1 - misses / observations:.2%}", flush=True)

    with (destination / "clean" / "trip_quality.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = ("split", "trip_id", "observations", "nodes", "arcs", "route_arcs",
                   "mean_candidates", "long_gaps", "reference_beyond_100m", "misses", "fit_ineligible")
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        for split, records in manifest["splits"].items():
            writer.writerows({"split": split, **record} for record in records)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def prepare_trip(archive, split, trip_id, config, tables, folder):
    """Validate one trip, generate candidates independently and align its labels."""
    trip = load_trip(archive, split, trip_id)
    validate_observation_order(trip)
    road = RoadGeometry.from_trip(trip, config)
    candidates = generate_candidates(road, config)
    # Fix candidates before opening the reference, including during preparation.
    route = load_reference_route(archive, trip)
    label = align_reference(road, route)
    matches = candidates.arc_ids == label.arcs[:, None]
    found = matches.any(axis=1)
    eligible = write_trip(tables, trip, road, route, label, found)
    diagnostics = {
        "trip_id": trip_id, "observations": len(label.arcs),
        "nodes": len(trip.nodes), "arcs": len(trip.arcs), "route_arcs": len(route),
        "mean_candidates": float(candidates.valid.sum() / len(label.arcs)),
        "long_gaps": int(np.count_nonzero(road.prior_gap > config.heading_gap_s)),
        "reference_beyond_100m": int(np.count_nonzero(label.distance_m > 100)),
        "misses": int(np.count_nonzero(~found)), "fit_ineligible": int(np.count_nonzero(~eligible)),
        "radius_grid": {},
    }
    for radius in (40, 70, 100, 150):
        if radius <= config.radius_m:
            available = candidates.features[:, :, 0] <= radius
            available[:, 0] = True  # Keep the nearest-arc fallback at each radius.
            diagnostics["radius_grid"][str(radius)] = int(
                np.count_nonzero(~(matches & available).any(axis=1))
            )
    # Fit uses fixed-width arrays. The CSV is the human-readable record and
    # the cache preserves which candidate slot each feature row belongs to.
    np.savez_compressed(
        folder / f"{trip_id}.npz",
        candidate_ids=candidates.arc_ids, features=candidates.features,
        reference_ids=label.arcs, reference_distances=label.distance_m,
        arc_nodes=road.arcs, gaps=road.prior_gap, fit_eligible=eligible,
    )
    return diagnostics


def load_prepared(root: Path, split: str, manifest: dict) -> list[PreparedTrip]:
    """Load only the declared development split, including older flat caches."""
    root = root / manifest.get("bundle", "")
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
