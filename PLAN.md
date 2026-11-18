# PathSnap implementation and commit plan

The five PathSnap handouts define the coursework deliverables. This plan proposes how the Section A team can implement and verify them in this repository. Each row is a reviewable milestone to commit to `main` after its checks pass.

## Milestones

| Step | Proposed commit | Lead | Reviewer | Implementation and checks |
| --- | --- | --- | --- | --- |
| 1 | `chore: bootstrap PathSnap and inventory student trips` | — | — | Completed in `df310bf` with Codex assistance. Verified the 80 training and 10 student-test trip layout against the supplied ZIP. |
| 2 | `feat: audit trip data and generate exploration maps` | Aditya | Ananya | Implement trip loading and validate numeric fields, coordinate ranges, timestamps, and graph references. Record per-trip GPS counts, bounds, sampling gaps, graph connectivity, and reference route continuity. Generate GPS plots and a Folium map with observed GPS, road geometry, and the reference route. |
| 3 | `feat: generate road candidates and align reference routes` | Akhila | Aditi | Compare candidate search radii in metric coordinates and measure reference coverage. Align observations to the ordered reference arcs, log ambiguous alignments, and make repeatable feature generation available for training and inference. Verify ID alignment and measure candidate miss rate by trip. |
| 4 | `feat: compare five map-matching classifiers` | Ananya | Aditya | Build the shared training and evaluation runner. Use trip-grouped cross-validation on the training split. Each model owner runs their experiments through the same runner. Record seeds, configurations, fold metrics, and timings, then compare the five models on student-test trips. |
| 5 | `feat: tune and export the selected classifier` | Ananya | Akhila | Choose the model from recorded results and justify the Bayesian search space and objective. Run group-aware cross-validation on training trips, save trial logs, refit, and compare tuned and untuned results on the same student-test trips. Verify that exported preprocessing and model artifacts reproduce saved predictions. |
| 6 | `feat: reconstruct routes with HMM and Viterbi` | Aditi | Akhila | Decode candidate arc states using classifier probabilities and directed road-network distances. Handle candidate misses, infeasible transitions, and long gaps. Compare classifier-only and decoded predictions on pointwise accuracy, continuity, and sequence-level metrics. Add the predicted route as a Folium layer for development trips. |
| 7a | `feat: serve prediction and scoring with FastAPI` | Akhila | Aditi | Serve inference and scoring on laptop 1. Agree on the response contract with the UI lead and verify that it returns the prediction, permitted metrics, and map data while keeping reference labels server-side. |
| 7b | `feat: connect the evaluator interface to the API` | Aditya | Ananya | Serve trip selection and results on laptop 2. Display the returned route and metrics, handle unavailable trips and API errors, and rehearse the LAN workflow with the API lead using development trips. |
| 8 | `docs: package experiment evidence and deployment steps` | Aditi | Aditya | Publish reproducible commands, the final model and configuration, comparison plots and source results, Folium map-generation code, and demo instructions. Assemble the report from each teammate’s sections. Coordinate the final rehearsal and verify all ten evaluator trips when organizers release them for evaluation. |

## Team contributions

Each teammate leads two work packages, reviews two other packages, runs at least one classifier experiment, and writes the report sections for their work. The lead owns implementation, verification, and the handoff. The reviewer reruns the relevant checks and records what they verified.

| Teammate | Classifier experiments | Report and demo contributions |
| --- | --- | --- |
| Aditya Shrivastav | Decision tree and random forest | Write the dataset audit and exploration sections. Supply Folium figures and UI screenshots, and demonstrate trip selection and results on laptop 2. |
| Akhila T | Histogram gradient boosting | Write the candidate generation, reference alignment, and backend sections. Supply API log screenshots and demonstrate server-side inference and scoring on laptop 1. |
| Ananya B R | Logistic regression | Write the shared evaluation protocol, model comparison, and tuning sections. Supply the comparison table, plots, trial logs, and model export checks. |
| Aditi Agarwal | Shallow MLP | Write the HMM and Viterbi section and provide the before-and-after route metrics. Assemble the PDF, check that its claims trace to experiment evidence, and coordinate the two-laptop rehearsal. |

Each model owner integrates their classifier into the shared runner, commits the code, configuration, run command, and recorded results, and writes that model’s report subsection. Ananya maintains the shared runner and consolidates the comparison. Record actual contributions with commit and result links as work lands. Revisit remaining effort after reference alignment and route decoding, when the team can see whether any work needs to move.

## Initial design decisions to validate

- **Reference alignment.** A `.route` file is an ordered list of traversed arcs. It has no timestamps and its length differs from the `.track` file in every supplied trip. Prototype a monotonic alignment using projected point-to-segment distance and the arc order, with explicit handling of long GPS gaps and repeated arcs. Check the label quality on mapped training examples and confirm the organizers’ scoring convention before finalizing pointwise evaluation.
- **Candidates and features.** Generate candidates independently from GPS and road geometry. Start with point-to-segment distance, GPS heading agreement when a heading is measurable, and properties of the directed segment. Measure candidate misses against the aligned reference separately so the classifier comparison accounts for the candidate generator’s failures.
- **Model comparison.** Keep all observations from a trip together in each fold and fit learned preprocessing inside each fold. Record CV macro F1, student-test precision, recall, macro F1, observation-level top-1, log loss where applicable, route continuity, and inference time. Use confusion matrices to diagnose candidate classification errors and observation-level top-1 across trips to guide model selection. Report the metrics and denominators explicitly when a reference arc is missing.
- **Deployment boundary.** The prediction pipeline takes GPS and road-network inputs. A separate scoring step on laptop 1 loads the evaluator reference after prediction. The evaluator UI displays the selected trip, returned route, and scoring results. The student-test split supports development comparison, and the ten reserved evaluator trips are used at final evaluation.

## Final submission

The public GitHub repository will contain the implementation, experiment evidence, model artifacts, and instructions. The separate technical PDF will cover dataset exploration, all model experiments, the Bayesian search, the HMM comparison, Folium visualisations, and deployment screenshots. Use only results generated by actual runs and retain enough evidence for the team to reproduce and explain them.
