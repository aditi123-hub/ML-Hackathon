"""Check the layout of the supplied PathSnap student dataset ZIP."""

import argparse
from collections import defaultdict
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile


EXPECTED_TRIPS = {"train": 80, "student_test": 10}
FILE_SUFFIXES = {".track", ".nodes", ".arcs", ".route"}


def inventory(archive_path: Path) -> dict[str, dict[str, set[str]]]:
    trips = {split: defaultdict(set) for split in EXPECTED_TRIPS}
    errors = []

    with ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            parts = PurePosixPath(entry.filename).parts
            if len(parts) < 2 or parts[0] != "Student folder" or parts[1] not in trips:
                continue

            split = parts[1]
            if len(parts) != 4:
                errors.append(f"Unexpected file path: {entry.filename}")
                continue

            trip_id, filename = parts[2:]
            path = PurePosixPath(filename)
            if path.stem != trip_id or path.suffix not in FILE_SUFFIXES:
                errors.append(f"Unexpected trip file: {entry.filename}")
                continue

            suffixes = trips[split][trip_id]
            if path.suffix in suffixes:
                errors.append(f"Duplicate file: {entry.filename}")
            if entry.file_size == 0:
                errors.append(f"Empty file: {entry.filename}")
            suffixes.add(path.suffix)

    for split, expected_count in EXPECTED_TRIPS.items():
        if len(trips[split]) != expected_count:
            errors.append(f"{split}: found {len(trips[split])} trips, expected {expected_count}")
        for trip_id, suffixes in sorted(trips[split].items()):
            missing = ", ".join(sorted(FILE_SUFFIXES - suffixes))
            if missing:
                errors.append(f"{split}/{trip_id}: missing {missing}")

    overlap = trips["train"].keys() & trips["student_test"].keys()
    if overlap:
        errors.append(f"Trip IDs appear in both splits: {', '.join(sorted(overlap))}")
    if errors:
        raise ValueError("\n".join(errors))

    return {split: dict(split_trips) for split, split_trips in trips.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to Student folder.zip")
    args = parser.parse_args()

    try:
        trips = inventory(args.archive)
    except (BadZipFile, OSError, ValueError) as error:
        parser.exit(1, f"Dataset inventory failed:\n{error}\n")

    for split, split_trips in trips.items():
        file_count = sum(len(suffixes) for suffixes in split_trips.values())
        print(f"{split}: {len(split_trips)} trips, {file_count} files")


if __name__ == "__main__":
    main()
