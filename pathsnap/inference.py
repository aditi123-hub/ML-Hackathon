"""Show live classifier predictions for one supplied training or student-test trip."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .alignment import align_reference
from .dataset import load_reference_route, load_trip
from .features import CandidateConfig, FEATURE_NAMES, RoadGeometry, generate_candidates
from .inventory import EXPECTED_TRIPS
from .metrics import predict_scores, top_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/classifier.joblib"))
    parser.add_argument("--split", choices=EXPECTED_TRIPS, default="student_test")
    parser.add_argument("--trip", default="00000065")
    parser.add_argument("--observation", type=int, default=21, help="One-based row to explain")
    parser.add_argument("--score", action="store_true", help="Score against the separate reference after prediction")
    args = parser.parse_args()

    # joblib can execute code during loading. Use only the shipped artifact or
    # a replacement produced by a trusted local training run.
    artifact = joblib.load(args.artifact)
    if tuple(artifact["feature_names"]) != FEATURE_NAMES:
        parser.error("The model expects a different feature schema")
    config = CandidateConfig(**artifact["candidate_config"])
    with ZipFile(args.archive) as archive, threadpool_limits(limits=3):
        trip = load_trip(archive, args.split, args.trip)
        if not 1 <= args.observation <= len(trip.observations):
            parser.error(f"Expected an observation between 1 and {len(trip.observations)}")
        road = RoadGeometry.from_trip(trip, config)
        candidates = generate_candidates(road, config)
        scores = predict_scores(artifact["model"], candidates.features, candidates.valid)
        predicted = top_one(scores, candidates.arc_ids)
        row = args.observation - 1
        ranked = np.argsort(-scores[row], kind="stable")
        output = {
            "trip": args.trip, "split": args.split,
            "model": artifact["model_name"], "variant": artifact["variant"],
            "observations": len(predicted), "example_observation": args.observation,
            "predicted_arc": int(predicted[row]),
            "top_candidates": [
                {
                    "arc_id": int(candidates.arc_ids[row, column]),
                    "match_score": round(float(scores[row, column]), 5),
                    "features": dict(zip(FEATURE_NAMES, candidates.features[row, column].astype(float).round(3).tolist())),
                }
                for column in ranked[:3] if candidates.valid[row, column]
            ],
        }
        if args.score:
            # The reference is first opened here, once predictions are fixed.
            reference = align_reference(road, load_reference_route(archive, trip))
            correct = int(np.count_nonzero(predicted == reference.arcs))
            output["reference_aligned_score"] = {
                "top1": correct / len(predicted), "correct": correct, "total": len(predicted),
                "reference_arc_for_example": int(reference.arcs[row]),
            }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
