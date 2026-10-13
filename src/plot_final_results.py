"""Plot final human-reviewed answer accuracy for Baseline and VersionRAG."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "dataset" / "evaluation_human_final.csv"
SUMMARY = ROOT / "dataset" / "final_accuracy_summary.csv"
FIGURE = ROOT / "dataset" / "final_accuracy.png"

GROUPS = {
    "Overall": None,
    "Content Retrieval": {
        "Content Retrieval",
        "Content Retrieval Complex",
        "Version-Specific Content Retrieval",
    },
    "Version Listing": {"Version Listing & Inquiry"},
    "Change Retrieval": {"Change Retrieval (e)", "Change Retrieval (i)"},
}
APPROACHES = ("Baseline", "VersionRAG")


def main():
    with INPUT.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 60:
        raise ValueError(f"Expected 60 questions, found {len(rows)}")
    if any(row[f"Score_{approach}_Human"] not in {"0", "1"}
           for row in rows for approach in APPROACHES):
        raise ValueError("All 120 human scores must be 0 or 1")

    grouped_types = set().union(*(types for types in GROUPS.values() if types))
    if {row["Type"] for row in rows} != grouped_types:
        raise ValueError("Question types do not match the chart groups")

    totals = {}
    for group, types in GROUPS.items():
        subset = rows if types is None else [row for row in rows if row["Type"] in types]
        totals[group] = {
            approach: (sum(row[f"Score_{approach}_Human"] == "1" for row in subset), len(subset))
            for approach in APPROACHES
        }

    with SUMMARY.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Approach"] + [f"{group} [%]" for group in GROUPS])
        for approach in APPROACHES:
            writer.writerow([approach] + [
                f"{100 * totals[group][approach][0] / totals[group][approach][1]:.2f}"
                for group in GROUPS
            ])

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 6.5))
    positions = np.arange(len(GROUPS))
    width = 0.34
    colors = {"Baseline": "#38527a", "VersionRAG": "#159b8e"}

    for offset, approach in zip((-width / 2, width / 2), APPROACHES):
        values = [100 * totals[group][approach][0] / totals[group][approach][1]
                  for group in GROUPS]
        bars = ax.bar(positions + offset, values, width, label=approach,
                      color=colors[approach])
        for bar, group in zip(bars, GROUPS):
            correct, count = totals[group][approach]
            ax.annotate(f"{100 * correct / count:.1f}%\n({correct}/{count})",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(positions, [f"{group}\n(n={totals[group]['Baseline'][1]})"
                              for group in GROUPS])
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20), [f"{value}%" for value in range(0, 101, 20)])
    ax.set_ylabel("Correct answers")
    ax.set_title("Answer accuracy by question group — final human review", pad=20)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.text(0.5, 0.03,
             "Content Retrieval includes basic, complex, and version-specific questions. "
             "Change Retrieval includes explicit and implicit questions.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGURE, dpi=220, facecolor="white")
    print(f"Saved {SUMMARY}")
    print(f"Saved {FIGURE}")


if __name__ == "__main__":
    main()
