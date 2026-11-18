"""Render first-evaluation model comparisons directly from the saved metrics."""

import argparse
import json
from pathlib import Path

import matplotlib


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

    # sklearn exposes an actual epoch history for the fitted MLP.
    loss = experiment["models"].get("mlp", {}).get("loss_curve", [])
    if loss:
        figure, axis = plt.subplots(figsize=(6.4, 3.3), layout="constrained")
        axis.plot(range(1, len(loss) + 1), loss, color="#1d54a5", linewidth=2)
        axis.set(xlabel="Epoch", ylabel="Training loss", title="MLP training loss on the final fit sample")
        axis.grid(alpha=0.2)
        figure.savefig(folder / "mlp_training_loss.png", dpi=160)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    render(parser.parse_args().results)


if __name__ == "__main__":
    main()
