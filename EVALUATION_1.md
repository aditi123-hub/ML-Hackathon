# PathSnap first evaluation

**UE24CS352A Machine Learning, Section A**  
Aditya Shrivastav · Akhila T · Ananya B R · Aditi Agarwal

Open [the live demo command](README.md#first-evaluation) first. Run the supplied student-test trip, show the feature values for observation 21, then open [the comparison figure](results/model_comparison.png). The saved model and evidence come from the supplied student archive. The reserved evaluator partition was outside this run.

## Data preprocessing and feature engineering (5 marks)

1. **Explain the joins.** A GPS observation has longitude, latitude, and time. The zero-based row of `.nodes` identifies a point. An `.arcs` row identifies a directed segment by its start and end node IDs. An ordered `.route` lists the traversed arc IDs. The loader validates numeric fields, ranges, and indices and preserves the original row IDs. The audit examined 80 training and 10 student-test trips, with 227,577 observations across 360 validated files.
2. **Show a sampling problem.** Open the Folium map and GPS plot produced by the audit in [README.md](README.md#dataset-exploration). Example training trip `00000014` has 448 GPS observations and 164 ordered reference arcs. Its largest sampling gap is 72 seconds. The code masks heading from GPS displacements across gaps longer than 20 seconds and masks small displacements below 2 metres.
3. **Generate candidates from geometry.** Project each trip to an azimuthal equidistant plane centered at its first GPS point. Query exact finite-segment distances with an R-tree. Keep at most the 20 nearest directed arcs within 150 m, using the nearest arc if the radius is empty. The training radius comparison at the same cap measured 88.79%, 89.21%, 89.49%, and 93.30% reference coverage for 40, 70, 100, and 150 m respectively. Student test coverage at 150 m was **19,150 of 19,160 (99.95%)**. See [the preparation manifest](results/preparation_manifest.json).
4. **Explain the 12 features.** Distance to the segment and excess distance over the nearest candidate quantify spatial fit. Arc length and the fractional projection along it describe geometry. The heading cosine distinguishes opposing directions, and its availability flag identifies when GPS heading is usable. Previous and next distances to the same arc bring in local trajectory evidence, with separate availability flags and clipped time gaps. Every feature can be computed from the trip inputs during prediction.
5. **Explain the labels.** The reference route has no observation timestamps. A monotone dynamic program aligns each GPS observation to an occurrence of a reference arc using distance, heading agreement, and a small penalty for forward jumps. The resulting arc ID labels candidates after retrieval. Repeat visits can have distinct route positions. The score measures agreement with these derived labels. Two training trips (`00000054` and `00000033`) account for 12,866 of the 13,965 training candidate misses, so reference quality and alignment on those trips deserve attention.

**Try it live.** Run the demo with and without `--score`. The same candidate IDs and scores appear. Scoring opens the reference only after inference. On student-test trip `00000065`, the shipped model matched 663 of 768 aligned observations.

## Model selection and training (5 marks)

**Controlled comparison.** We compared logistic regression, decision tree, random forest, histogram gradient boosting, and a shallow MLP with the same candidate features and fixed three-fold `GroupKFold` splits. A trip stays in one fold. The seed is 352. For each fit we sample up to 256 observations per fit trip, retain up to eight nearest candidates plus any retrieved positive, and score every retrieved validation or student-test candidate. The exported fit used 154,836 candidate rows. The linear and neural models fit their feature scalers inside their own training pipelines.

| Model | CV top-1 | CV macro F1 | Student top-1 | Student macro F1 | Student log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 72.50% | 85.56% | 83.31% | 88.34% | 0.0759 |
| Decision tree | 72.93% | 87.71% | 84.58% | 91.10% | 0.0772 |
| Random forest | 73.68% | 88.55% | 84.82% | 91.45% | 0.0539 |
| Histogram gradient boosting | 73.80% | 88.57% | 85.17% | 91.45% | 0.0529 |
| Shallow MLP | 73.48% | 88.46% | 84.78% | 91.00% | 0.0549 |
| **Tuned histogram gradient boosting** | **73.92%** | **88.58%** | **85.25%** | **91.57%** | **0.0527** |

**Why these models.** The regularized logistic model provides a linear reference. The depth-limited tree learns thresholds and interactions and uses Gini impurity. The forest averages 64 trees to reduce variance and improved student-test log loss over the single tree. Histogram boosting adds trees to reduce binary log loss and had the highest grouped CV top-1. The scaled 32/16-unit MLP also trains on log loss, learned nonlinear interactions, and reached the fixed 38-epoch cap with a convergence warning. Its measured epoch history is in [the MLP loss plot](results/mlp_training_loss.png). Exact settings and precision, recall, confusion matrices, timings, training sample scores, and per-fold results are in [experiments.json](results/experiments.json).

**How Bayesian tuning was done.** Optuna TPE ran 10 trials on the same grouped training folds, with 3 startup trials. It searched tree count, leaves, learning rate, minimum leaf size, and L2 regularization. The best trial used 200 iterations, 42 leaves, a learning rate near 0.0581, 17 samples per leaf, and L2 near 2.438. It had the best grouped CV top-1 and was refit on the training trips. The student comparison improved from 85.17% to **85.25% (16,334 / 19,160)**, below the original 96–97% target. The candidate miss rate is 10 / 19,160. The primary remaining error in that split is candidate ranking.

**Viva distinctions.** Macro F1 averages the binary match and non-match F1 scores across candidate rows. Observation top-1 makes one choice per GPS row, so observations with a missing reference candidate count as incorrect. Report the tested split and metric each time. The per-observation classifier is the scope of this checkpoint. Sequential decoding, the final two-laptop demonstration, and the consolidated PDF belong to the next phase.

**Disclosure.** Codex assisted with implementation, experiment execution, and these notes. Reproduce and inspect the evidence before presenting it. Record the team's actual review, experiment reruns, and changes as described in [COMMIT_PLAN.md](COMMIT_PLAN.md).
