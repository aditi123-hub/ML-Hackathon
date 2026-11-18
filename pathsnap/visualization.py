"""Figures for reviewing GPS observations against the supplied road geometry."""

from html import escape
from itertools import pairwise
from pathlib import Path

import folium
import matplotlib

from .dataset import Trip


GPS_COLOR = "#1d54a5"
REFERENCE_COLOR = "#157053"
ROAD_COLOR = "#7a8491"


def write_trajectory_figure(trip: Trip, reference: list[int], output: Path) -> None:
    # Render a file without requiring a desktop session.
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection

    longitudes, latitudes, timestamps = zip(*trip.observations)
    road_segments = [(trip.nodes[a], trip.nodes[b]) for a, b in trip.arcs]
    figure, (route_axis, time_axis) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    route_axis.add_collection(
        LineCollection(road_segments, colors=ROAD_COLOR, linewidths=0.55, alpha=0.5, label="Road network"),
        autolim=False,
    )
    route_axis.add_collection(
        LineCollection(
            [road_segments[arc_id] for arc_id in reference],
            colors=REFERENCE_COLOR,
            linewidths=2,
            label="Reference arcs",
        )
    )
    route_axis.scatter(longitudes, latitudes, s=5, color=GPS_COLOR, alpha=0.8, label="GPS observations")
    route_axis.autoscale_view()
    route_axis.set(xlabel="Longitude (°)", ylabel="Latitude (°)", title="Trajectory and road network")
    route_axis.ticklabel_format(useOffset=False)
    route_axis.legend(loc="best", fontsize=8)

    intervals = [later - earlier for earlier, later in pairwise(timestamps)]
    time_axis.axhline(1, color=ROAD_COLOR, linestyle="--", linewidth=1, label="1-second step")
    time_axis.scatter(range(2, len(timestamps) + 1), intervals, s=6, color=GPS_COLOR)
    time_axis.set(
        xlabel="Observation index (1-based)", ylabel="Time step (s, symlog)", title="GPS sampling intervals"
    )
    time_axis.set_yscale("symlog", linthresh=1)
    time_axis.legend(fontsize=8)
    figure.suptitle(f"{trip.split} · {trip.trip_id}")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def write_trip_map(trip: Trip, reference: list[int], output: Path) -> None:
    gps = [(point.latitude, point.longitude) for point in trip.observations]
    map_view = folium.Map(location=gps[0], tiles="OpenStreetMap", prefer_canvas=True, max_zoom=19)
    map_view.get_root().title = escape(f"PathSnap {trip.split} · {trip.trip_id}")

    folium.PolyLine(
        [[trip.nodes[a][::-1], trip.nodes[b][::-1]] for a, b in trip.arcs],
        color=ROAD_COLOR,
        weight=1,
        opacity=0.55,
        interactive=False,
        smooth_factor=0,
    ).add_to(folium.FeatureGroup(name="Road network", control=True).add_to(map_view))
    # Each reference arc has its own segment, including on either side of a discontinuity.
    reference_segments = [
        [trip.nodes[node][::-1] for node in trip.arcs[arc_id]]
        for arc_id in reference
    ]
    folium.PolyLine(
        reference_segments, color=REFERENCE_COLOR, weight=4, opacity=0.9, interactive=False, smooth_factor=0
    ).add_to(folium.FeatureGroup(name="Reference arcs", control=True).add_to(map_view))

    gps_layer = folium.FeatureGroup(name="GPS observations", control=True).add_to(map_view)
    for index, (location, observation) in enumerate(zip(gps, trip.observations), start=1):
        folium.CircleMarker(
            location=location,
            radius=3,
            stroke=False,
            fill=True,
            fill_color=GPS_COLOR,
            fill_opacity=0.85,
            tooltip=f"Observation {index} · {observation.timestamp:g} s",
        ).add_to(gps_layer)

    focus = gps + [point for segment in reference_segments for point in segment]
    latitudes, longitudes = zip(*focus)
    map_view.fit_bounds(
        [(min(latitudes), min(longitudes)), (max(latitudes), max(longitudes))], padding=(24, 24), max_zoom=17
    )
    folium.LayerControl(collapsed=False).add_to(map_view)
    map_view.save(output)
