import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEED_ROLLOUT_RE = re.compile(
    r"Seed Long Rollout Validation Loss \((?P<steps>\d+) steps\): "
    r"(?P<loss>[0-9.]+) \| min rho=(?P<min_rho>[0-9.eE+-]+) \| min T=(?P<min_t>[0-9.eE+-]+)"
)
ROLLOUT_RE = re.compile(
    r"Long Rollout Validation Loss \((?P<steps>\d+) steps\): "
    r"(?P<loss>[0-9.]+) \| min rho=(?P<min_rho>[0-9.eE+-]+) \| min T=(?P<min_t>[0-9.eE+-]+)"
)
EPOCH_RE = re.compile(r"Epochs:.*?\|\s*(?P<epoch>\d+)/(?P<total>\d+)\s*\[")


@dataclass
class RunProgress:
    label: str
    log_path: str
    rollout_steps: int
    epochs: List[int]
    rollout_losses: List[float]
    min_rhos: List[float]
    min_temperatures: List[float]

    @property
    def best_index(self) -> int:
        return int(np.argmin(self.rollout_losses))

    @property
    def best_epoch(self) -> int:
        return self.epochs[self.best_index]

    @property
    def best_loss(self) -> float:
        return self.rollout_losses[self.best_index]

    @property
    def latest_epoch(self) -> int:
        return self.epochs[-1]

    @property
    def latest_loss(self) -> float:
        return self.rollout_losses[-1]


def parse_run_arg(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--run entries must look like LABEL=/path/to/log.log, got {value!r}"
        )
    label, log_path = value.split("=", 1)
    label = label.strip()
    log_path = log_path.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"Run label cannot be empty: {value!r}")
    if not log_path:
        raise argparse.ArgumentTypeError(f"Run log path cannot be empty: {value!r}")
    return label, log_path


def parse_reference_arg(value: str) -> Tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--reference-loss entries must look like LABEL=0.012393, got {value!r}"
        )
    label, raw_value = value.split("=", 1)
    label = label.strip()
    raw_value = raw_value.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"Reference label cannot be empty: {value!r}")
    try:
        numeric_value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Reference loss must be numeric, got {raw_value!r}"
        ) from exc
    return label, numeric_value


def load_progress(label: str, log_path: str) -> RunProgress:
    epochs: List[int] = []
    rollout_losses: List[float] = []
    min_rhos: List[float] = []
    min_temperatures: List[float] = []
    rollout_steps = None
    last_epoch = None

    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                last_epoch = int(epoch_match.group("epoch"))

            seed_match = SEED_ROLLOUT_RE.search(line)
            if seed_match:
                rollout_steps = int(seed_match.group("steps"))
                epochs.append(0)
                rollout_losses.append(float(seed_match.group("loss")))
                min_rhos.append(float(seed_match.group("min_rho")))
                min_temperatures.append(float(seed_match.group("min_t")))
                continue

            rollout_match = ROLLOUT_RE.search(line)
            if rollout_match is None:
                continue
            if last_epoch is None:
                raise ValueError(
                    f"Found rollout validation line before any epoch marker in {log_path}"
                )
            if epochs and last_epoch == epochs[-1]:
                continue

            rollout_steps = int(rollout_match.group("steps"))
            epochs.append(last_epoch)
            rollout_losses.append(float(rollout_match.group("loss")))
            min_rhos.append(float(rollout_match.group("min_rho")))
            min_temperatures.append(float(rollout_match.group("min_t")))

    if not rollout_losses:
        raise ValueError(f"No rollout validation entries found in {log_path}")
    if rollout_steps is None:
        raise ValueError(f"Could not infer rollout step count from {log_path}")

    return RunProgress(
        label=label,
        log_path=log_path,
        rollout_steps=rollout_steps,
        epochs=epochs,
        rollout_losses=rollout_losses,
        min_rhos=min_rhos,
        min_temperatures=min_temperatures,
    )


def plot_progress(
    runs: List[RunProgress],
    reference_losses: List[Tuple[str, float]],
    output_path: str,
    title: str,
    temperature_floor: Optional[float],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), squeeze=False)
    ax_loss = axes[0][0]
    ax_temp = axes[0][1]
    colors = plt.get_cmap("tab10").colors

    for index, run in enumerate(runs):
        color = colors[index % len(colors)]
        epochs = np.asarray(run.epochs)
        losses = np.asarray(run.rollout_losses)
        best_so_far = np.minimum.accumulate(losses)
        min_temperatures = np.asarray(run.min_temperatures)

        ax_loss.plot(
            epochs,
            losses,
            color=color,
            linewidth=1.6,
            alpha=0.35,
            marker="o",
            markersize=3,
        )
        ax_loss.plot(
            epochs,
            best_so_far,
            color=color,
            linewidth=2.2,
            label=f"{run.label} best={run.best_loss:.6f} @ e{run.best_epoch}",
        )

        ax_temp.plot(
            epochs,
            min_temperatures,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3,
            label=f"{run.label} latest={run.min_temperatures[-1]:.4f}",
        )

    for label, value in reference_losses:
        ax_loss.axhline(
            value,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label=f"{label} = {value:.6f}",
        )

    if temperature_floor is not None:
        ax_temp.axhline(
            temperature_floor,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label=f"T floor = {temperature_floor:g}",
        )

    rollout_steps = runs[0].rollout_steps
    ax_loss.set_title(f"{rollout_steps}-step rollout validation loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.25)
    ax_loss.legend(fontsize=8)

    ax_temp.set_title(f"Minimum temperature over {rollout_steps}-step rollout")
    ax_temp.set_xlabel("Epoch")
    ax_temp.set_ylabel("Min temperature")
    ax_temp.grid(alpha=0.25)
    ax_temp.legend(fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_summary(runs: List[RunProgress], reference_losses: List[Tuple[str, float]]) -> None:
    print("Parsed stage-2 rollout progress:")
    for run in runs:
        seed_loss = run.rollout_losses[0]
        seed_min_t = run.min_temperatures[0]
        latest_min_t = run.min_temperatures[-1]
        rel_improvement = (seed_loss - run.best_loss) / seed_loss if seed_loss else 0.0
        print(
            f"- {run.label}: seed={seed_loss:.6f}, best={run.best_loss:.6f} @ epoch {run.best_epoch}, "
            f"latest={run.latest_loss:.6f} @ epoch {run.latest_epoch}, "
            f"improvement={100.0 * rel_improvement:.1f}%, "
            f"minT {seed_min_t:.6f} -> {latest_min_t:.6f}"
        )
    for label, value in reference_losses:
        print(f"- reference {label}: {value:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot stage-2 long-rollout validation progress from SOD training logs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run_arg,
        help="LABEL=/path/to/log.log (repeat for multiple runs)",
    )
    parser.add_argument(
        "--reference-loss",
        action="append",
        default=[],
        type=parse_reference_arg,
        help="Optional horizontal reference line, e.g. BarrierV3=0.012393",
    )
    parser.add_argument(
        "--temperature-floor",
        type=float,
        default=None,
        help="Optional horizontal line for minimum rollout temperature.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="SOD Case 2 stage-2 rollout progress",
    )
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    runs = [load_progress(label, log_path) for label, log_path in args.run]
    rollout_steps = {run.rollout_steps for run in runs}
    if len(rollout_steps) != 1:
        raise ValueError(
            "All runs must use the same long-rollout validation horizon; "
            f"found {sorted(rollout_steps)}"
        )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plot_progress(
        runs=runs,
        reference_losses=args.reference_loss,
        output_path=args.output,
        title=args.title,
        temperature_floor=args.temperature_floor,
    )
    print_summary(runs, args.reference_loss)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
