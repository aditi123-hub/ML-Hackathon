# PathSnap implementation and commit plan

The five PathSnap handouts define the coursework deliverables. This plan proposes how the Section A team can implement and verify them in this repository. Each row is a reviewable milestone to commit to `main` after its checks pass.

## Milestones

| Step | Proposed commit | Implementation and checks |
| --- | --- | --- |
| 1 | `chore: bootstrap PathSnap and inventory student trips` | Initialize the repository and verify the 80 training and 10 student-test trip layout against the supplied ZIP. |
| 2 | `feat: audit trip data and generate exploration maps` | Implement trip loading and validate numeric fields, coordinate ranges, timestamps, and graph references. Record per-trip GPS counts, bounds, sampling gaps, graph connectivity, and reference route continuity. Generate GPS plots and a Folium map with observed GPS, road geometry, and the reference route. |
| 3 | `feat: generate road candidates and align reference routes` | Compare candidate search radii in metric coordinates and measure reference coverage. Align observations to the ordered reference arcs, log ambiguous alignments, and make repeatable feature generation available for training and inference. Verify ID alignment and measure candidate miss rate by trip. |
| 4 | `feat: compare five map-matching classifiers` | Start with logistic regression, decision tree, random forest, histogram gradient boosting, and a shallow MLP. Use trip-grouped cross-validation on the training split. Record seeds, configurations, fold metrics, and timings, then compare the five models on student-test trips. |
| 5 | `feat: tune and export the selected classifier` | Choose the model from recorded results and justify the Bayesian search space and objective. Run group-aware cross-validation on training trips, save trial logs, refit, and compare tuned and untuned results on the same student-test trips. Verify that exported preprocessing and model artifacts reproduce saved predictions. |
| 6 | `feat: reconstruct routes with HMM and Viterbi` | Decode candidate arc states using classifier probabilities and directed road-network distances. Handle candidate misses, infeasible transitions, and long gaps. Compare classifier-only and decoded predictions on pointwise accuracy, continuity, and sequence-level metrics. Add the predicted route as a Folium layer for development trips. |
| 7 | `feat: serve the predictor and connect the evaluator UI` | Serve inference and scoring with FastAPI on laptop 1 and trip selection and results on laptop 2. Return the prediction, permitted metrics, and map data. Rehearse the LAN workflow with development trips and verify API logs and response contents. |
| 8 | `docs: package experiment evidence and deployment steps` | Publish reproducible commands, the final model and configuration, comparison plots and source results, Folium map-generation code, and demo instructions. Prepare the separate consolidated PDF with experiment evidence, AI-use disclosure, and screenshots. Verify laptop-to-laptop inference for all ten reserved trips when organizers release them for evaluation. |

## Initial design decisions to validate

- **Reference alignment.** A `.route` file is an ordered list of traversed arcs. It has no timestamps and its length differs from the `.track` file in every supplied trip. Prototype a monotonic alignment using projected point-to-segment distance and the arc order, with explicit handling of long GPS gaps and repeated arcs. Check the label quality on mapped training examples and confirm the organizers’ scoring convention before finalizing pointwise evaluation.
- **Candidates and features.** Generate candidates independently from GPS and road geometry. Start with point-to-segment distance, GPS heading agreement when a heading is measurable, and properties of the directed segment. Measure candidate misses against the aligned reference separately so the classifier comparison accounts for the candidate generator’s failures.
- **Model comparison.** Keep all observations from a trip together in each fold and fit learned preprocessing inside each fold. Record CV macro F1, student-test precision, recall, macro F1, observation-level top-1, log loss where applicable, route continuity, and inference time. Use confusion matrices to diagnose candidate classification errors and observation-level top-1 across trips to guide model selection. Report the metrics and denominators explicitly when a reference arc is missing.
- **Deployment boundary.** The prediction pipeline takes GPS and road-network inputs. A separate scoring step on laptop 1 loads the evaluator reference after prediction. The evaluator UI displays the selected trip, returned route, and scoring results. The student-test split supports development comparison, and the ten reserved evaluator trips are used at final evaluation.

## Final submission

The public GitHub repository will contain the implementation, experiment evidence, model artifacts, and instructions. The separate technical PDF will cover dataset exploration, all model experiments, the Bayesian search, the HMM comparison, Folium visualisations, and deployment screenshots. Use only results generated by actual runs and retain enough evidence for the team to reproduce and explain them.
