"""Trip storage and response assembly for the evaluator's server."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

import numpy as np

from .dataset import load_reference_route, load_trip
from .inventory import FILE_SUFFIXES, inventory
from .scoring import score_predictions


EVALUATOR_TRIP_COUNT = 10


@dataclass(frozen=True)
class TripStore:
    """List requestable trip IDs with the input files held on the API host."""

    path: Path
    partition: str
    trip_ids: tuple[str, ...]

    @classmethod
    def from_student_archive(cls, path):
        return cls(path, "student_test", tuple(sorted(inventory(path)["student_test"])))

    @classmethod
    def from_evaluator_directory(cls, path):
        """Check filenames for the extracted final partition without reading rows."""
        identifiers = []
        for folder in sorted(path.iterdir()):
            if not folder.is_dir() or not re.fullmatch(r"[0-9]{8}", folder.name):
                continue
            missing = [
                suffix for suffix in sorted(FILE_SUFFIXES)
                if not (folder / f"{folder.name}{suffix}").is_file()
            ]
            if missing:
                raise ValueError(f"Evaluator trip {folder.name} is missing {', '.join(missing)}")
            identifiers.append(folder.name)
        if len(identifiers) != EVALUATOR_TRIP_COUNT:
            raise ValueError(f"Expected {EVALUATOR_TRIP_COUNT} evaluator trip folders, found {len(identifiers)}")
        return cls(path, "evaluator", tuple(identifiers))

    @contextmanager
    def open(self):
        # Open a separate ZIP handle per request so threaded requests can read
        # independently. An extracted evaluator partition needs no ZIP layout.
        if self.partition == "student_test":
            with ZipFile(self.path) as archive:
                yield archive
        else:
            yield DirectoryReader(self.path)


@dataclass(frozen=True)
class DirectoryReader:
    """Translate the shared dataset loader's member name into an extracted path."""

    root: Path

    def open(self, member):
        folder, filename = PurePosixPath(member).parts[-2:]
        return (self.root / folder / filename).open("rb")


def evaluation_result(store, matcher, trip_id):
    """Run prediction first, then score it and package only permitted map layers."""
    with store.open() as source:
        trip = load_trip(source, store.partition, trip_id)
        prediction = matcher.predict(trip)
        # The inference method receives only the GPS and road network. Open
        # the route here and return aggregates without per-fix reference IDs.
        scores, _ = score_predictions(
            prediction, load_reference_route(source, trip), matcher.decoder.max_gap_s,
        )

    candidates = prediction.candidates
    # Candidate roads have already been filtered against GPS. Deduplicate IDs
    # so frequently encountered roads are not repeated in the map payload.
    nearby_arcs = np.unique(candidates.arc_ids[candidates.valid]).tolist()
    return {
        "trip_id": trip_id,
        "partition": store.partition,
        "model": matcher.name,
        "variant": matcher.variant,
        "elapsed_s": prediction.total_seconds,
        "scores": scores,
        "map": {
            # GeoJSON coordinate order, longitude then latitude. Restrict the
            # base geometry to roads near GPS fixes to keep responses small.
            "gps": [(point.longitude, point.latitude) for point in trip.observations],
            "roads": [[trip.nodes[node] for node in trip.arcs[arc]] for arc in nearby_arcs],
            "routes": [
                # Contiguous blocks share junctions. Emit the first source,
                # then each target to make one polyline without duplicated joints.
                [trip.nodes[trip.arcs[block[0]][0]], *(trip.nodes[trip.arcs[arc][1]] for arc in block)]
                for block in prediction.decoded.blocks
            ],
        },
    }
