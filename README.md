# PathSnap

PES University · UE24CS352A Machine Learning · Section A

Aditya Shrivastav, Akhila T, Ananya B R, and Aditi Agarwal

PathSnap is a hackathon project to match noisy GPS observations to a directed road network and reconstruct vehicle routes. See [the implementation and commit plan](PLAN.md) for the path from dataset exploration to the two-laptop demo.

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

## Initial data inspection

| File | Contents | How to join |
| --- | --- | --- |
| `.track` | Longitude, latitude, and timestamp in seconds | One row per GPS observation, in recorded order |
| `.nodes` | Longitude and latitude | Zero-based row index is the node ID |
| `.arcs` | Directed from-node and to-node IDs | Zero-based row index is the arc ID |
| `.route` | Sequence of reference arc IDs | Join by arc ID within the same trip |

An initial scan of the supplied archive found 208,417 training observations and 19,160 student-test observations. The longest consecutive timestamp gaps were 153,064 seconds on training trip `00000090` and 210 seconds on student-test trip `00000065`. Across both splits, 14 consecutive reference-arc pairs lacked a direct endpoint match. The audit planned in [PLAN.md](PLAN.md) will reproduce these checks and investigate their impact on alignment and route continuity.

## AI assistance

Codex assisted with initial archive inspection, the inventory command, and the project plan.
