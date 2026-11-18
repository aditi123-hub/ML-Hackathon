"""Render classifier and route comparisons directly from the saved metrics."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np


def render(folder: Path):
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    experiment = json.loads((folder / "experiments.json").read_text(encoding="utf-8"))
    names = {
        "logistic": "Logistic regression",
        "tree": "Decision tree",
        "forest": "Random forest",
        "boosting": "Histogram boosting",
        "mlp": "Shallow MLP",
    }
    models = list(experiment["models"].items())
    labels = [names[name] for name, _ in models]
    source = [
        ("Grouped CV macro F1", [row["cv"]["mean_macro_f1"] for _, row in models]),
        ("Student-test top-1", [row["student_test"]["top1"] for _, row in models]),
        ("Student-test macro F1", [row["student_test"]["macro_f1"] for _, row in models]),
        ("Student-test log loss", [row["student_test"]["log_loss"] for _, row in models]),
    ]
    fig, panels = plt.subplots(2, 2, figsize=(10, 6.2), layout="constrained")
    for index, (axis, (title, scores)) in enumerate(zip(panels.flat, source)):
        bars = axis.barh(labels, scores, color="#1d54a5")
        axis.invert_yaxis()
        axis.set_title(title, loc="left")
        axis.spines[["right", "top", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.grid(axis="x", alpha=0.18)
        if index < 3:
            axis.set_xlim(0, 1.08)
            axis.xaxis.set_major_formatter(PercentFormatter(1))
            axis.bar_label(bars, labels=[f"{score:.1%}" for score in scores], padding=3, fontsize=8)
        else:
            axis.set_xlim(0, max(scores) * 1.23)
            axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig.savefig(folder / "model_comparison.png", dpi=160)
    plt.close(fig)

    # Render the real recorded candidate confusion counts. Normalize color
    # by true class so the majority of nonmatching arcs does not hide misses.
    if "tuned" in experiment:
        models.append((f"{experiment['selected']} tuned", experiment["tuned"]))
    figure, panels = plt.subplots(2, 3, figsize=(9.3, 5.6), layout="constrained")
    for axis, (name, row) in zip(panels.flat, models):
        counts = np.asarray(row["student_test"]["confusion"])
        denominators = counts.sum(axis=1, keepdims=True)
        rates = np.divide(counts, denominators, out=np.zeros(counts.shape), where=denominators > 0)
        axis.imshow(rates, cmap="Blues", vmin=0, vmax=1)
        for truth, predicted in np.ndindex(counts.shape):
            axis.text(
                predicted, truth, f"{counts[truth, predicted]:,}\n{rates[truth, predicted]:.1%}",
                ha="center", va="center", color="white" if rates[truth, predicted] > 0.6 else "#152742",
            )
        axis.set_title(names.get(name, name))
        axis.set_xticks([0, 1], ["Other", "Match"])
        axis.set_yticks([0, 1], ["Other", "Match"])
    for axis in list(panels.flat)[len(models):]:
        axis.set_visible(False)
    figure.supxlabel("Predicted candidate class")
    figure.supylabel("Reference candidate class")
    figure.savefig(folder / "confusion_matrices.png", dpi=160)
    plt.close(figure)

    # sklearn exposes an actual epoch history for the fitted MLP.
    loss = experiment["models"].get("mlp", {}).get("loss_curve", [])
    if loss:
        figure, axis = plt.subplots(figsize=(6.4, 3.3), layout="constrained")
        axis.plot(range(1, len(loss) + 1), loss, color="#1d54a5", linewidth=2)
        axis.set(xlabel="Epoch", ylabel="Training loss", title="MLP training loss on the final fit sample")
        axis.grid(alpha=0.2)
        figure.savefig(folder / "mlp_training_loss.png", dpi=160)
        plt.close(figure)

    learning_report = folder / "mlp_learning_curve.json"
    if learning_report.exists():
        # Both series are evaluated after each real epoch on the same
        # frozen grouped split. The diagnostic is independent of export.
        history = json.loads(learning_report.read_text(encoding="utf-8"))["history"]
        epochs = [row["epoch"] for row in history]
        figure, axis = plt.subplots(figsize=(6.4, 3.3), layout="constrained")
        for key, label, color in (
            ("training_log_loss", "Fit candidates", "#1d54a5"),
            ("validation_log_loss", "Held-out trips", "#a83e18"),
        ):
            axis.plot(epochs, [row[key] for row in history], color=color, label=label)
        axis.set(xlabel="Epoch", ylabel="Candidate log loss", title="MLP learning curve on grouped fold 1")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.savefig(folder / "mlp_grouped_loss.png", dpi=160)
        plt.close(figure)

    route_report = folder / "route_comparison.json"
    if route_report.exists():
        routes = json.loads(route_report.read_text(encoding="utf-8"))["trips"]
        figure, axis = plt.subplots(figsize=(10, 5.5), layout="constrained")
        rows = range(len(routes))
        styles = (
            ("classifier", "Classifier", "#566a7e", "o"),
            ("viterbi", "Viterbi", "#1d54a5", "s"),
            ("decoded", "Final route", "#a83e18", "D"),
        )
        for stage, label, color, marker in styles:
            axis.scatter(
                [row[stage]["top1"] for row in routes], rows, c=color, marker=marker, label=label,
            )
        axis.axvline(0.96, color="#7a8491", linestyle="--", label="96% target")
        axis.set_yticks(list(rows), [row["trip_id"] for row in routes])
        axis.invert_yaxis()
        axis.set(xlabel="Observation-level top-1", ylabel="Student-test trip")
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", alpha=0.2)
        axis.legend(loc="lower left")
        figure.savefig(folder / "route_comparison.png", dpi=160)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/current"))
    render(parser.parse_args().results)


if __name__ == "__main__":
    main()
