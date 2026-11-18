# PathSnap

PES University · UE24CS352A Machine Learning · Section A

Aditya Shrivastav, Akhila T, Ananya B R, and Aditi Agarwal

PathSnap is a hackathon project to match noisy GPS observations to a directed road network and reconstruct vehicle routes. See [the implementation and commit plan](PLAN.md) for the path from dataset exploration to the two-laptop demo.

## First evaluation

The [viva guide](EVALUATION_1.md) covers preprocessing, features, training, and model selection. The [commit handoff](COMMIT_PLAN.md) assigns files and verification to each teammate. This checkpoint runs the per-observation classifier on the supplied student data.

From the repository root on Windows, use Python 3.11–3.13. The packaged model was created with Python 3.13 and the pinned dependencies below.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$studentZip = Join-Path $env:USERPROFILE 'Downloads/Student folder.zip'
.\.venv\Scripts\python.exe -m pathsnap.inference $studentZip --trip 00000065 --observation 21 --score
```

The demo prints the three highest-scoring candidates for GPS observation 21. `--score` aligns the local reference after prediction and reports correct and total GPS observations. Omit it to show the independent prediction path. Load the included joblib artifact or one from a trusted training run.

Reproduce the cached features, five grouped comparisons, and ten Bayesian trials:

```powershell
.\.venv\Scripts\python.exe -m pathsnap.preparation $studentZip --radius 150 --limit 20
.\.venv\Scripts\python.exe -m pathsnap.training --trials 10
.\.venv\Scripts\python.exe -m pathsnap.experiment_plots
```

The cache goes under `outputs/prepared/`. Training writes `results/experiments.json` with fold, fit, and trial details, `results/model_comparison.csv` with the comparison summary, and `artifacts/classifier.joblib` with the selected fitted model and its feature configuration. `results/preparation_manifest.json` records per-trip coverage and the supplied archive hash for the bundled experiment. Copy it from `outputs/prepared/manifest.json` after a rerun. For a single baseline, use `--model tree` or another model name with separate `--results` and `--artifacts` directories.

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
