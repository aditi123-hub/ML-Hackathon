"""Run the five-model comparison and seeded Bayesian tuning on training trips."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from .features import FEATURE_NAMES
from .metrics import evaluate
from .models import FACTORIES, SEARCH_SPACES
from .preparation import load_prepared


SEED = 352


def fitting_rows(batches, max_observations, seed):
    """Bound fit cost while keeping every training trip and hard negatives.

    Sample whole observations within each trip with a seeded RNG. For each,
    retain up to eight near candidates and any aligned positive already in the
    retrieved set. Validation and student testing score every retrieved candidate.
    """
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for batch in batches:
        size = len(batch.reference_ids)
        rows = np.arange(size)
        if size > max_observations:
            rows = np.sort(rng.choice(rows, max_observations, replace=False))
        candidates = batch.candidate_ids[rows]
        target = candidates == batch.reference_ids[rows, None]
        keep = (candidates >= 0) & ((np.arange(candidates.shape[1]) < 8) | target)
        features.append(batch.features[rows][keep])
        labels.append(target[keep])
    return np.concatenate(features), np.concatenate(labels)


def make_folds(batches, n_splits, max_observations, seed):
    identifiers = [batch.trip_id for batch in batches]
    folds = []
    for fit_indices, valid_indices in GroupKFold(n_splits=n_splits).split(identifiers, groups=identifiers):
        training = [batches[index] for index in fit_indices]
        validation = [batches[index] for index in valid_indices]
        folds.append((fitting_rows(training, max_observations, seed), validation))
    return folds


def cross_validate(factory, params, folds, seed):
    outcomes = []
    for (features, labels), validation in folds:
        model = factory(seed, **params)
        started = perf_counter()
        model.fit(features, labels)
        elapsed = perf_counter() - started
        report = evaluate(model, validation)
        outcomes.append(report | {
            "training_s": elapsed,
            "training_rows": len(labels),
            "validation_trips": [trip.trip_id for trip in validation],
        })
    return {
        "folds": outcomes,
        "mean_top1": float(np.mean([outcome["top1"] for outcome in outcomes])),
        "mean_macro_f1": float(np.mean([outcome["macro_f1"] for outcome in outcomes])),
    }


def fit_and_compare(factory, params, full, testing, seed):
    features, labels = full
    model = factory(seed, **params)
    started = perf_counter()
    model.fit(features, labels)
    elapsed = perf_counter() - started
    classifier = model[-1] if hasattr(model, "steps") else model
    probabilities = model.predict_proba(features)[:, 1]
    return model, {
        "configuration": classifier.get_params(),
        "training_s": elapsed,
        "training_rows": len(labels),
        "fit_macro_f1": float(f1_score(labels, probabilities >= 0.5, average="macro")),
        "fit_log_loss": float(log_loss(labels, probabilities)),
        "student_test": evaluate(model, testing),
        "loss_curve": getattr(classifier, "loss_curve_", []),
    }


def tune(name, folds, count, seed):
    """TPE models the distribution of better and worse observed settings."""
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=3)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        settings = SEARCH_SPACES[name](trial)
        result = cross_validate(FACTORIES[name], settings, folds, seed)
        trial.set_user_attr("cv", result)
        return result["mean_top1"]

    study.optimize(objective, n_trials=count, gc_after_trial=True)
    return {
        "objective": "Mean observation-level top-1 across held-out training-trip folds",
        "sampler": "Optuna TPE (three startup trials)",
        "best_trial": study.best_trial.number,
        "best_params": study.best_params,
        "space": {key: str(value) for key, value in study.best_trial.distributions.items()},
        "trials": [
            {"trial": trial.number, "params": trial.params, "top1": trial.value,
             "cv": trial.user_attrs["cv"], "seconds": trial.duration.total_seconds()}
            for trial in study.trials
        ],
        "best_cv": study.best_trial.user_attrs["cv"],
    }


def write_comparison(path, result):
    keys = ["model", "cv_macro_f1", "cv_top1", "student_top1", "student_macro_f1",
            "precision", "recall", "log_loss", "direct_continuity", "inference_s", "training_s"]
    models = dict(result["models"])
    if "tuned" in result:
        models[f"{result['selected']}_tuned"] = result["tuned"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(keys)
        for name, row in models.items():
            cv, test = row["cv"], row["student_test"]
            writer.writerow([
                name, cv["mean_macro_f1"], cv["mean_top1"], test["top1"], test["macro_f1"],
                test["precision"], test["recall"], test["log_loss"], test["direct_continuity"],
                test["inference_s"], row["training_s"],
            ])


def train(args):
    manifest = json.loads((args.prepared / "manifest.json").read_text(encoding="utf-8"))
    if tuple(manifest["features"]) != FEATURE_NAMES:
        raise ValueError("Feature schema changed. Re-run pathsnap.preparation.")
    training = load_prepared(args.prepared, "train", manifest)
    testing = load_prepared(args.prepared, "student_test", manifest)
    if {trip.trip_id for trip in training} & {trip.trip_id for trip in testing}:
        raise ValueError("Training and student-test trip IDs overlap")
    folds = make_folds(training, args.folds, args.fit_observations, args.seed)
    full = fitting_rows(training, args.fit_observations, args.seed)
    del training

    result = {
        "seed": args.seed, "folds": args.folds,
        "fit_observations_per_trip": args.fit_observations,
        "candidate_config": manifest["candidate_config"], "feature_names": FEATURE_NAMES,
        "archive_sha256": manifest["archive_sha256"],
        "selection_metric": "Grouped CV observation top-1, with macro F1 as a tie-breaker",
        "models": {},
    }
    args.results.mkdir(parents=True, exist_ok=True)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    # Model artifacts are ordinary trusted joblib files produced by this run.
    # Keep each temporary fit separate so the final selection uses recorded CV.
    scratch_models = args.prepared / "baseline_models"
    scratch_models.mkdir(exist_ok=True)
    with threadpool_limits(limits=args.threads):
        names = (args.model,) if args.model else FACTORIES
        for name in names:
            print(f"Fitting {name} in {args.folds} grouped folds", flush=True)
            factory = FACTORIES[name]
            cv = cross_validate(factory, {}, folds, args.seed)
            model, record = fit_and_compare(factory, {}, full, testing, args.seed)
            result["models"][name] = record | {"cv": cv}
            joblib.dump(model, scratch_models / f"{name}.joblib", compress=3)
            print(f"{name}: CV top-1 {cv['mean_top1']:.2%}, student top-1 {record['student_test']['top1']:.2%}", flush=True)

        selected = max(result["models"], key=lambda name: (
            result["models"][name]["cv"]["mean_top1"], result["models"][name]["cv"]["mean_macro_f1"]
        ))
        result["selected"] = selected
        exported = joblib.load(scratch_models / f"{selected}.joblib")
        result["exported_variant"] = "untuned"
        if args.trials and not args.model:
            study = tune(selected, folds, args.trials, args.seed)
            tuned, record = fit_and_compare(FACTORIES[selected], study["best_params"], full, testing, args.seed)
            result["tuning"] = study
            result["tuned"] = record | {"cv": study["best_cv"]}
            if study["best_cv"]["mean_top1"] >= result["models"][selected]["cv"]["mean_top1"]:
                exported = tuned
                result["exported_variant"] = "tuned"

    artifact = {
        "model": exported, "model_name": selected, "variant": result["exported_variant"],
        "feature_names": FEATURE_NAMES, "candidate_config": manifest["candidate_config"],
    }
    joblib.dump(artifact, args.artifacts / "classifier.joblib", compress=3)
    # Confirm that serializing the fitted pipeline preserves inference.
    restored = joblib.load(args.artifacts / "classifier.joblib")
    probe = full[0][:64]
    np.testing.assert_array_equal(exported.predict_proba(probe), restored["model"].predict_proba(probe))
    (args.results / "experiments.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_comparison(args.results / "model_comparison.csv", result)
    print(f"Exported {selected} ({result['exported_variant']})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("outputs/prepared"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fit-observations", type=int, default=256)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model", choices=FACTORIES, help="Reproduce a single baseline. Use distinct output folders.")
    args = parser.parse_args()
    if args.trials < 0 or min(args.fit_observations, args.threads) < 1 or not 2 <= args.folds <= 80:
        parser.error("Expected nonnegative trials, positive threads and fit observations, and 2 to 80 folds")
    train(args)


if __name__ == "__main__":
    main()
