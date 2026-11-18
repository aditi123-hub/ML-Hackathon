# PathSnap

PES University · UE24CS352A Machine Learning · Section A

Aditya Shrivastav, Akhila T, Ananya B R, and Aditi Agarwal

PathSnap is a hackathon project to match noisy GPS observations to a directed road network and reconstruct vehicle routes. See [the implementation and commit plan](PLAN.md) for the path from dataset exploration to the two-laptop demo.

Start with the [code guide](docs/CODE_GUIDE.md) to find any implementation and rehearse the coordinates-to-metres example. The [guidelines audit](docs/GUIDELINES_AUDIT.md) shows the evidence for each requirement. The [final commit plan](docs/FINAL_COMMIT_PLAN.md) assigns two commits and two reviews to each teammate.

## Current results

The tuned histogram boosting model and route decoder matched **18,855 of 19,160 observations (98.41%)** against the aligned references across all ten student-test trips. Each observation counts once, including ten candidate misses. The supplied references list arcs in order and carry no per-observation timestamps, so the pointwise metric uses monotone reference alignment.

| Stage | Correct | Observation top-1 | Mean trip sequence F1 |
| --- | ---: | ---: | ---: |
| Tuned classifier | 17,504 | 91.36% | 83.47% |
| Classifier + Viterbi | 18,015 | 94.02% | 98.37% |
| Final route with refined observation assignments | 18,855 | 98.41% | 98.37% |

The last pass assigns each GPS observation to a candidate on its **predicted** route block using distance, heading and traversal order. Across the ten trips, final accuracy ranged from 94.57% to 100.00%. Trips `00000009` and `00000091` fell below 96%. The student-test results measure development performance. The separately released evaluator partition is for the final demo.

[Five-model comparison](results/current/model_comparison.csv) · [Per-trip route scores](results/current/route_comparison.csv) · [Per-trip accuracy plot](results/current/route_comparison.png) · [Training log](results/current/training.log) · [Folium route map](results/current/route-00000065.html)

## Clean data and training

Use Python 3.11–3.13 from the repository root. The commands below use the original student ZIP.

```powershell
python -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install -r requirements.txt
$studentZip = Join-Path $env:USERPROFILE 'Downloads/Student folder.zip'
& $python -m pathsnap.preparation $studentZip
& $python -m pathsnap.training --threads 5 --trials 10
& $python -m pathsnap.mlp_diagnostic
& $python -m pathsnap.route_eval $studentZip --calibrate
& $python -m pathsnap.experiment_plots
```

Preparation prints the CSV directory under `outputs/prepared/bundles/<bundle>/clean/`. Within each split, read `trip_id` as text to retain leading zeroes and join keys within a trip. All record IDs start at zero.

| CSV | Key and contents |
| --- | --- |
| `observations.csv` | Trip and observation ID, longitude, latitude, timestamp and elapsed gap |
| `nodes.csv` | Trip and node ID, longitude and latitude |
| `arcs.csv` | Trip and directed arc ID, source and target node IDs |
| `reference_route.csv` | Trip and route position, reference arc ID |
| `aligned_labels.csv` | Trip and observation ID, aligned route position and arc ID, residual distance, candidate availability and fit eligibility |
| `trip_quality.csv` | Split and trip, row counts, long gaps, missed candidates and labels excluded from fitting |

The loader checks numeric fields, coordinate ranges and graph references. Preparation checks increasing timestamps and preserves gaps and row IDs. The fitting sample skips observations with an aligned arc farther than 60 m away or absent from the candidate set. Validation and student-test scoring keep those observations. The same preparation pass writes the compact NPZ candidate cache used for training.

Training prints each fold and the Bayesian trial settings and metrics to the terminal and writes the same output to `results/current/training.log`. `experiments.json` includes fold membership, confusion matrices, timings and parameters. The five model factories live under [pathsnap/models](pathsnap/models). [fitting.py](pathsnap/fitting.py) applies the shared trip-grouped protocol. [inference.py](pathsnap/inference.py) loads the exported classifier, its feature schema and the adjacent decoder settings. Feature scaling for logistic regression and MLP is fitted inside each fold. The decoder settings were selected on a seeded 12-trip training sample.

To inspect one trip and see all stages in the terminal:

```powershell
& $python -m pathsnap.inference $studentZip --trip 00000065 --observation 21 --decode --score
```

## Two-laptop evaluator demo

On laptop 1, install `requirements.txt` and start FastAPI. For a rehearsal, use the student ZIP. The API logs each request, timing and final score in this terminal.

```powershell
& $python -m pathsnap.api --student-zip $studentZip --host 0.0.0.0 --port 8000
```

For the final run, use the released evaluator archive on laptop 1 only. The extracted `evaluator_test` directory contains ten immediate trip folders with `.track`, `.nodes`, `.arcs` and `.route` files. The commands below extract to Git-ignored storage. Skip extraction if the folder is already present.

```powershell
$evaluatorZip = Join-Path $env:USERPROFILE 'Downloads/evaluator_test.zip'
Expand-Archive -LiteralPath $evaluatorZip -DestinationPath 'outputs/final-evaluation'
& $python -m pathsnap.api --evaluator-dir 'outputs/final-evaluation/evaluator_test' --host 0.0.0.0 --port 8000
```

On laptop 2, run the UI from the repository using laptop 1's LAN IP and open [the local evaluator UI](http://127.0.0.1:8001). For a single-computer rehearsal, use `http://127.0.0.1:8000` as the API origin. Allow port 8000 on laptop 1's private-network firewall if required by your network setup.

```powershell
python -m pathsnap.ui --api http://192.168.1.10:8000 --port 8001
```

Select a trip and click **Evaluate trip**. The UI shows the predicted route, GPS fixes, nearby roads and aggregate scores. Its map works with geometry returned by the API and requires no external map tiles. The API opens the reference file after prediction and sends only the aggregate scores. The HTTP contract exposes readiness at `GET /healthz`, available trips at `GET /api/trips` and evaluation at `POST /api/trips/{trip_id}/predict`.

## First evaluation

The [viva guide](EVALUATION_1.md) covers preprocessing, features, training, and model selection. The [commit handoff](COMMIT_PLAN.md) assigns files and verification to each teammate. This checkpoint runs the per-observation classifier on the supplied student data.

From the repository root on Windows, use Python 3.11–3.13. The packaged model was created with Python 3.13 and the pinned dependencies below.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$studentZip = Join-Path $env:USERPROFILE 'Downloads/Student folder.zip'
.\.venv\Scripts\python.exe -m pathsnap.inference $studentZip --artifact artifacts/classifier.joblib --trip 00000065 --observation 21 --score
```

The demo prints the three highest-scoring candidates for GPS observation 21. `--score` aligns the local reference after prediction and reports correct and total GPS observations. Omit it to show the independent prediction path. Load the included joblib artifact or one from a trusted training run.

The original checkpoint artifacts and metrics are in `artifacts/classifier.joblib` and `results/`. The updated preparation, training and scoring commands are under [Clean data and training](#clean-data-and-training). To run one baseline, use `--model tree` or another model name with separate `--results` and `--artifacts` directories.

## Check the student dataset

From the repository root, run the inventory with Python 3.10 or later. It uses the standard library and reads the supplied ZIP directly.

```powershell
python -m pathsnap.inventory "$env:USERPROFILE/Downloads/Student folder.zip"
```

The inventory checks the expected trip counts, split overlap, and the four trip files in each folder. Expected output for the supplied archive:

```text
train: 80 trips, 320 files
student_test: 10 trips, 40 files
```

Keep the original archives and extracted trip data local. The student archive provides the training and development splits. The ten reserved evaluator trips are for the final demo.

## Dataset exploration

| File | Contents | How to join |
| --- | --- | --- |
| `.track` | Longitude, latitude, and timestamp in seconds | One row per GPS observation, in recorded order |
| `.nodes` | Longitude and latitude | Zero-based row index is the node ID |
| `.arcs` | Directed from-node and to-node IDs | Zero-based row index is the arc ID |
| `.route` | Sequence of reference arc IDs | Join by arc ID within the same trip |

Run the full audit and generate the example plot and map from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-eda.txt
.\.venv\Scripts\python.exe -m pathsnap.explore "$env:USERPROFILE/Downloads/Student folder.zip"
```

The command writes `outputs/eda/audit.json` with per-trip file sizes, GPS bounds, sampling intervals, road connectivity, and reference route diagnostics. `train-00000014.png` plots the road network, observations, reference arcs, and sampling intervals for the example trip. Open `train-00000014.html` in a browser to toggle the Folium layers. The interactive map loads browser assets and basemap tiles online. Use `--map-split student_test --map-trip 00000065` to plot another trip and `--output` to change the destination.

### Observed results

| Measure | Training | Student test |
| --- | ---: | ---: |
| Trips | 80 | 10 |
| GPS observations | 208,417 | 19,160 |
| Exactly one-second steps | 201,048 of 208,337 (96.5%) | 18,886 of 19,150 (98.6%) |
| Longest gap | 153,064 s (`00000090`) | 210 s (`00000065`) |
| Trips with multiple weak road components | 79 | 10 |
| Reference arc pairs without a direct join | 13 | 1 |

All 360 files passed validation for field count, numeric values, coordinate ranges, and in-range IDs where applicable. Timestamps had zero repeated or backward steps. The loader preserves the recorded order and row IDs and loads reference arc sequences separately. Weak components treat arcs as undirected for the connectivity count. The reference continuity check compares the previous arc's end node with the next arc's start node in the directed graph.

## AI assistance

Codex assisted with preprocessing, implementation, experiment execution, visualizations, and the handoff. The retained scores are from actual runs on the supplied student archive. Each teammate should reproduce the experiments they present and record their actual review and changes.
