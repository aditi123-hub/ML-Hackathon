"""Run the PathSnap inference and scoring API on laptop 1."""

import argparse
import logging
from pathlib import Path
from zipfile import BadZipFile

from fastapi import FastAPI, HTTPException
from threadpoolctl import threadpool_limits
import uvicorn

from .inference import DEFAULT_ARTIFACT, Matcher
from .service import TripStore, evaluation_result


log = logging.getLogger("uvicorn.error")


def create_app(store: TripStore, matcher: Matcher) -> FastAPI:
    """Bind a frozen model and the server-side trip source to the HTTP endpoints."""
    app = FastAPI(title="PathSnap evaluation API")

    @app.get("/healthz")
    def health():
        return {"status": "ready"}

    @app.get("/api/trips")
    def trips():
        return {"partition": store.partition, "trips": store.trip_ids}

    @app.post("/api/trips/{trip_id}/predict")
    def predict(trip_id: str):
        # Membership is checked before any file is opened. A trip ID is
        # a catalog key and never an arbitrary path supplied by a client.
        if trip_id not in store.trip_ids:
            raise HTTPException(status_code=404, detail="Trip is unavailable in this partition.")
        try:
            result = evaluation_result(store, matcher, trip_id)
        except (ValueError, OSError, BadZipFile):
            log.exception("Could not evaluate %s/%s", store.partition, trip_id)
            raise HTTPException(status_code=500, detail="The server could not evaluate this trip.")
        decoded = result["scores"]["decoded"]
        log.info(
            "Evaluated %s/%s in %.2fs, %d/%d decoded observations (%.2f%%)",
            store.partition, trip_id, result["elapsed_s"], decoded["correct"],
            result["scores"]["observations"], 100 * decoded["top1"],
        )
        return result

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Select a single source at startup so rehearsal and final requests
    # cannot mix partitions within a running service.
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--student-zip", type=Path, help="Use the ten student-test trips for rehearsal")
    source.add_argument("--evaluator-dir", type=Path, help="Directory containing the released evaluator trip folders")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--decoder-config", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    store = (
        TripStore.from_student_archive(args.student_zip) if args.student_zip
        else TripStore.from_evaluator_directory(args.evaluator_dir)
    )
    matcher = Matcher.load(args.artifact, args.decoder_config)
    with threadpool_limits(limits=args.threads):
        uvicorn.run(create_app(store, matcher), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
