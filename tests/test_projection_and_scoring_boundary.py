"""Check coordinate units, legacy feature parity and reference access ordering."""

from contextlib import contextmanager
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import unittest

import numpy as np
from pyproj import Geod, Proj

from pathsnap.dataset import Observation, Trip
from pathsnap.features import CandidateConfig
from pathsnap.geometry import RoadGeometry
from pathsnap.inference import DEFAULT_ARTIFACT, Matcher
from pathsnap.projection import project_trip_coordinates
from pathsnap.service import evaluation_result


class ProjectionTests(unittest.TestCase):
    def test_projected_radial_length_is_in_metres(self):
        # Cross the date line. A raw longitude difference would span nearly
        # 360 degrees, while the measured east step is only about 200 metres.
        gps = np.array([[179.999, 12], [-179.999, 12]])
        points, nodes = project_trip_coordinates(gps, gps)
        _, _, metres = Geod(ellps="WGS84").inv(*gps[0], *gps[1])
        np.testing.assert_array_equal(points, nodes)
        np.testing.assert_allclose(points[0], [0, 0], atol=1e-9)
        self.assertGreater(points[1, 0], 0)
        self.assertAlmostEqual(np.linalg.norm(points[1]), metres, places=6)

    def test_projection_refactor_matches_previously_exported_schema(self):
        coordinates = np.array([[77.601, 12.91], [77.604, 12.92], [77.613, 12.897]])
        before = Proj(proj="aeqd", lon_0=coordinates[0, 0], lat_0=coordinates[0, 1], ellps="WGS84")
        expected = np.column_stack(before(coordinates[:, 0], coordinates[:, 1], errcheck=True))
        projected, _ = project_trip_coordinates(coordinates, coordinates)
        np.testing.assert_array_equal(projected, expected)

    def test_snap_fraction_is_relative_to_the_directed_segment(self):
        trip = Trip("test", "00000001", [Observation(0, 0, 1)], [(0, 0), (0.001, 0)], [(0, 1), (1, 0)])
        road = RoadGeometry.from_trip(trip, CandidateConfig())
        # Use exact metre vectors so the fraction and residual are simple to verify.
        road = replace(road, starts=np.array([[0., 0.], [100., 0.]]),
                       vectors=np.array([[100., 0.], [-100., 0.]]), lengths=np.array([100., 100.]))
        distances, fractions = road.distance_and_fraction(np.array([25., 10.]), np.array([0, 1]))
        np.testing.assert_allclose(distances, [10, 10])
        np.testing.assert_allclose(fractions, [0.25, 0.75])


class ReferenceBoundaryTests(unittest.TestCase):
    def test_reference_is_opened_after_prediction_and_never_sent_as_map_geometry(self):
        # Keep this fixture wholly synthetic. The one available arc gives
        # every observation a deterministic candidate and route.
        files = {
            ".track": b"0.0000 0 1\n0.0001 0 2\n",
            ".nodes": b"0 0\n0.001 0\n",
            ".arcs": b"0 1\n",
            ".route": b"0\n",
        }
        events = []
        matcher = Matcher.load(Path(DEFAULT_ARTIFACT))

        class Reader:
            def open(self, member):
                suffix = Path(member).suffix
                events.append(suffix)
                return BytesIO(files[suffix])

        class Store:
            partition = "test"

            @contextmanager
            def open(self):
                yield Reader()

        class ObservedMatcher:
            name, variant, decoder = matcher.name, matcher.variant, matcher.decoder

            @staticmethod
            def predict(trip):
                self.assertEqual(events, [".track", ".nodes", ".arcs"])
                result = matcher.predict(trip)
                events.append("predicted")
                return result

        result = evaluation_result(Store(), ObservedMatcher(), "00000001")
        self.assertEqual(events, [".track", ".nodes", ".arcs", "predicted", ".route"])
        self.assertEqual(set(result["map"]), {"gps", "roads", "routes"})
        self.assertEqual(result["scores"]["decoded"]["correct"], 2)


if __name__ == "__main__":
    unittest.main()
