import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def load_results(json_path: Path):
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected JSON file to contain a list of checkpoint results.")
    return data


def prepare_class_series(results):
    class_series = defaultdict(list)
    time_labels = []

    for idx, entry in enumerate(results):
        checkpoint_path = entry.get("checkpoint", f"checkpoint_{idx}")
        time_label = Path(checkpoint_path).name
        time_labels.append(time_label)
        for stat in entry.get("class_stats", []):
            class_name = stat.get("class_name", f"class_{stat.get('class_id', 'unknown')}")
            speed = float(stat.get("speed_mean", 0.0))
            iou = float(stat.get("iou_mean", 0.0))
            class_series[class_name].append(
                {
                    "time_idx": idx,
                    "time_label": time_label,
                    "speed": speed,
                    "iou": iou,
                }
            )
    return class_series, time_labels


def plot_series(class_series, time_labels, output_path: Path):
    if not class_series:
        print("No class statistics found. Aborting plot.")
        return

    num_times = len(time_labels)
    time_colors = plt.cm.viridis(np.linspace(0, 1, num_times)) if num_times > 0 else ["#333333"]
    class_colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(class_series))))

    fig, ax = plt.subplots(figsize=(12, 8))

    for idx, (class_name, series) in enumerate(sorted(class_series.items())):
        sorted_series = sorted(series, key=lambda x: x["time_idx"])
        speeds = [item["speed"] for item in sorted_series]
        ious = [item["iou"] for item in sorted_series]
        time_indices = [item["time_idx"] for item in sorted_series]
        ax.plot(
            speeds,
            ious,
            label=class_name,
            color=class_colors[idx % len(class_colors)],
            linewidth=1.5,
            alpha=0.8,
        )
        for s, i, t_idx in zip(speeds, ious, time_indices):
            color = time_colors[t_idx % len(time_colors)]
            ax.scatter(
                s,
                i,
                color=color,
                edgecolor="black",
                s=60,
                zorder=5,
            )

    ax.set_xlabel("Average Speed", fontsize=12)
    ax.set_ylabel("Mean IoU", fontsize=12)
    ax.set_title("Class-wise Mean IoU vs Average Speed Across Checkpoints", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)

    class_legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="Classes",
        fontsize=9,
    )
    ax.add_artist(class_legend)

    time_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=time_colors[i % len(time_colors)],
            markeredgecolor="black",
            markersize=8,
            label=time_labels[i],
        )
        for i in range(len(time_labels))
    ]
    if time_handles:
        ax.legend(
            handles=time_handles,
            title="Checkpoints (time)",
            loc="upper left",
            bbox_to_anchor=(1.02, 0.4),
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize class-wise speed vs mIoU over checkpoints.")
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("/workspace/gan_seg/outputs/mask_sequence_metrics.json"),
        help="Path to the JSON file containing evaluation metrics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/gan_seg/outputs/plots/mask_miou_vs_speed.png"),
        help="Output path for the generated plot.",
    )
    args = parser.parse_args()

    results = load_results(args.json)
    class_series, time_labels = prepare_class_series(results)
    plot_series(class_series, time_labels, args.output)


if __name__ == "__main__":
    main()
