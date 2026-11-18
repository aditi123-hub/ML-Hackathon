"""Load the row-indexed GPS and road-network files for one PathSnap trip."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from math import isfinite
from typing import NamedTuple, TypeVar
from zipfile import ZipFile


Coordinate = tuple[float, float]
Arc = tuple[int, int]
Number = TypeVar("Number", int, float)


class Observation(NamedTuple):
    longitude: float
    latitude: float
    timestamp: float


@dataclass
class Trip:
    split: str
    trip_id: str
    observations: list[Observation]
    nodes: list[Coordinate]
    arcs: list[Arc]


def trip_member(split: str, trip_id: str, suffix: str) -> str:
    return f"Student folder/{split}/{trip_id}/{trip_id}{suffix}"


def _rows(
    archive: ZipFile, member: str, width: int, parse: Callable[[bytes], Number]
) -> Iterator[tuple[int, tuple[Number, ...]]]:
    try:
        stream = archive.open(member)
    except KeyError as error:
        raise ValueError(f"Missing trip file: {member}") from error

    with stream:
        for line_number, line in enumerate(stream, start=1):
            columns = line.split()
            if len(columns) != width:
                raise ValueError(f"{member}:{line_number}: expected {width} fields, found {len(columns)}")
            try:
                values = tuple(parse(column) for column in columns)
            except ValueError as error:
                raise ValueError(f"{member}:{line_number}: invalid numeric field") from error
            yield line_number, values
        if not stream.tell():
            raise ValueError(f"{member}: empty file")


def _check_coordinate(longitude: float, latitude: float, location: str) -> None:
    if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
        raise ValueError(f"{location}: invalid longitude or latitude")


def load_trip(archive: ZipFile, split: str, trip_id: str) -> Trip:
    """Read inputs used for prediction. IDs are zero-based positions in the lists."""
    track_file = trip_member(split, trip_id, ".track")
    observations = []
    for line, (longitude, latitude, timestamp) in _rows(archive, track_file, 3, float):
        _check_coordinate(longitude, latitude, f"{track_file}:{line}")
        if not isfinite(timestamp):
            raise ValueError(f"{track_file}:{line}: timestamp must be finite")
        observations.append(Observation(longitude, latitude, timestamp))

    nodes_file = trip_member(split, trip_id, ".nodes")
    nodes = []
    for line, (longitude, latitude) in _rows(archive, nodes_file, 2, float):
        _check_coordinate(longitude, latitude, f"{nodes_file}:{line}")
        nodes.append((longitude, latitude))

    arcs_file = trip_member(split, trip_id, ".arcs")
    arcs = []
    for line, (source, target) in _rows(archive, arcs_file, 2, int):
        if not (0 <= source < len(nodes) and 0 <= target < len(nodes)):
            raise ValueError(f"{arcs_file}:{line}: arc endpoint outside node table")
        arcs.append((source, target))

    return Trip(split, trip_id, observations, nodes, arcs)


def load_reference_route(archive: ZipFile, trip: Trip) -> list[int]:
    """Read the ordered reference arc IDs for development analysis or scoring."""
    route_file = trip_member(trip.split, trip.trip_id, ".route")
    route = []
    for line, (arc_id,) in _rows(archive, route_file, 1, int):
        if not 0 <= arc_id < len(trip.arcs):
            raise ValueError(f"{route_file}:{line}: reference ID outside arc table")
        route.append(arc_id)
    return route
