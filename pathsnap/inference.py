"""Predict candidate arcs and routes from GPS and the directed road network."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .dataset import Trip, load_reference_route, load_trip
from .decoder import DecodedRoute, DecoderConfig, decode
from .road_network import RoadNetwork
from .features import CandidateConfig, Candidates, generate_candidates
from .geometry import RoadGeometry
from .inventory import EXPECTED_TRIPS
from .metrics import predict_scores, top_one
from .scoring import score_predictions


DEFAULT_ARTIFACT = Path("artifacts/current/classifier.joblib")


@dataclass(frozen=True)
class Matcher:
    """Production inference contract. The input Trip has no reference labels."""

    model: object
    name: str
    variant: str
    candidates: CandidateConfig
    decoder: DecoderConfig

    @classmethod
    def load(cls, path: Path, decoder_path: Path | None = None):
        # joblib loading executes Python code. Use a model from a trusted
        # local training run or the artifact shipped with the repository.
        artifact = joblib.load(path)
        config = CandidateConfig.from_schema(artifact["candidate_config"], artifact["feature_names"])
        if decoder_path is None:
            sibling = path.with_name("decoder.json")
            decoder_path = sibling if sibling.is_file() else None
        settings = json.loads(decoder_path.read_text(encoding="utf-8")) if decoder_path else {}
        return cls(artifact["model"], artifact["model_name"], artifact["variant"], config, DecoderConfig(**settings))

    def predict(self, trip: Trip, reconstruct: bool = True):
        """Convert geometry, score candidate pairs and optionally decode the route."""
        started = perf_counter()
        road = RoadGeometry.from_trip(trip, self.candidates)
        candidates = generate_candidates(road, self.candidates)
        scores = predict_scores(self.model, candidates.features, candidates.valid)
        # Top-1 picks one arc for each GPS fix independently of route continuity.
        classifier_arcs = top_one(scores, candidates.arc_ids)
        classifier_seconds = perf_counter() - started
        decoded = decode(RoadNetwork(road), candidates, scores, self.decoder) if reconstruct else None
        return Prediction(road, candidates, scores, classifier_arcs, decoded, classifier_seconds, perf_counter() - started)


@dataclass(frozen=True)
class Prediction:
    """Reference-free outputs used by the offline scorer and the API."""

    road: RoadGeometry
    candidates: Candidates
    scores: np.ndarray
    classifier_arcs: np.ndarray
    decoded: DecodedRoute | None
    classifier_seconds: float
    total_seconds: float


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--decoder-config", type=Path)
    parser.add_argument("--split", choices=EXPECTED_TRIPS, default="student_test")
    parser.add_argument("--trip", default="00000065")
    parser.add_argument("--observation", type=int, default=21, help="One-based row to explain")
    parser.add_argument("--score", action="store_true", help="Score against the separate reference after prediction")
    parser.add_argument("--decode", action="store_true", help="Also reconstruct a connected route with Viterbi")
    parser.add_argument("--explain-projection", action="store_true", help="Show GPS degrees and projected metres for the example")
    args = parser.parse_args()

    matcher = Matcher.load(args.artifact, args.decoder_config)
    with ZipFile(args.archive) as archive, threadpool_limits(limits=3):
        trip = load_trip(archive, args.split, args.trip)
        if not 1 <= args.observation <= len(trip.observations):
            parser.error(f"Expected an observation between 1 and {len(trip.observations)}")
        prediction = matcher.predict(trip, reconstruct=args.decode)
        candidates, scores, predicted = prediction.candidates, prediction.scores, prediction.classifier_arcs
        row = args.observation - 1
        ranked = np.argsort(-scores[row], kind="stable")
        output = {
            "trip": args.trip, "split": args.split,
            "model": matcher.name, "variant": matcher.variant,
            "observations": len(predicted), "example_observation": args.observation,
            "predicted_arc": int(predicted[row]),
            "top_candidates": [
                {
                    "arc_id": int(candidates.arc_ids[row, column]),
                    "match_score": round(float(scores[row, column]), 5),
                    "features": dict(zip(
                        matcher.candidates.feature_names, candidates.features[row, column].astype(float).round(3).tolist(),
                    )),
                }
                for column in ranked[:3] if candidates.valid[row, column]
            ],
        }
        if args.explain_projection:
            fix = trip.observations[row]
            origin = trip.observations[0]
            east_m, north_m = prediction.road.points[row]
            output["projection"] = {
                "method": "WGS84 azimuthal equidistant",
                "origin_longitude_latitude_deg": [origin.longitude, origin.latitude],
                "fix_longitude_latitude_deg": [fix.longitude, fix.latitude],
                "fix_x_east_y_north_m": [float(east_m), float(north_m)],
                "distance_from_origin_m": float(np.hypot(east_m, north_m)),
            }
        if prediction.decoded is not None:
            output["decoded_arc"] = int(prediction.decoded.observation_arcs[row])
            output["predicted_route_blocks"] = prediction.decoded.blocks
        if args.score:
            # The reference is first opened here, once predictions are fixed.
            scores, aligned = score_predictions(
                prediction, load_reference_route(archive, trip), matcher.decoder.max_gap_s,
            )
            output["reference_aligned_score"] = {
                "top1": scores["classifier"]["top1"], "correct": scores["classifier"]["correct"],
                "total": scores["observations"], "reference_arc_for_example": int(aligned.arcs[row]),
            }
            if prediction.decoded is not None:
                output["route_comparison"] = scores
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
