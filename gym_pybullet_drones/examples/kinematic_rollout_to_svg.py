"""Convert rollout Logger-format .npz files into kinematic rollout SVG plots.

This script searches a results tree for ``kinematic_rollout_*.npz`` archives
saved by the examples (e.g. ``pid-vertical.py``, ``MultiModelLearn.py``),
reconstructs a minimal ``Logger`` instance from each archive, and calls
``Logger.plot()`` to generate the same kinematic rollout figure in SVG format.

Each SVG is written alongside its source .npz, using the same basename, e.g.:

    kinematic_rollout_cf2x.npz -> kinematic_rollout_cf2x.svg

Usage (from repo root, with ``PYTHONPATH=.`` or editable install):

    $ python gym_pybullet_drones/examples/kinematic_rollout_to_svg.py \
        --root results

By default, existing SVGs are left untouched; use ``--overwrite true`` to
regenerate them.
"""

import argparse
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from gym_pybullet_drones.utils.Logger import Logger

# Mapping from human-friendly channel names to Logger.states indices and labels.
CHANNELS: dict[str, tuple[int, str]] = {
    "x": (0, "x (m)"),
    "y": (1, "y (m)"),
    "z": (2, "z (m)"),
    "vx": (3, "vx (m/s)"),
    "vy": (4, "vy (m/s)"),
    "vz": (5, "vz (m/s)"),
    "roll": (6, "roll (rad)"),
    "pitch": (7, "pitch (rad)"),
    "yaw": (8, "yaw (rad)"),
    "wx": (9, "wx (rad/s)"),
    "wy": (10, "wy (rad/s)"),
    "wz": (11, "wz (rad/s)"),
}


def _load_logger_from_npz(path: str) -> Logger:
    """Create a Logger instance populated from a rollout npz file."""
    with np.load(path) as data:
        timestamps = data["timestamps"]
        states = data["states"]
        controls = data["controls"]
        logging_freq_hz = int(data["logging_freq_hz"])

    # Ensure shapes are (num_drones, *, timesteps)
    if timestamps.ndim == 1:
        timestamps = timestamps.reshape(1, -1)
    if states.ndim == 2:
        states = states.reshape(1, states.shape[0], states.shape[1])
    if controls.ndim == 2:
        controls = controls.reshape(1, controls.shape[0], controls.shape[1])

    num_drones = states.shape[0]
    timesteps = states.shape[2]

    logger = Logger(
        logging_freq_hz=logging_freq_hz,
        output_folder=os.path.dirname(path),
        num_drones=num_drones,
        duration_sec=0,
        colab=False,
    )

    # Overwrite preallocated arrays with the rollout data
    logger.timestamps = timestamps
    logger.states = states
    logger.controls = controls
    logger.counters = np.full(num_drones, timesteps, dtype=float)

    return logger


def _discover_rollouts(root: str) -> list[str]:
    """Recursively find kinematic_rollout_*.npz files under root."""
    matches: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.startswith("kinematic_rollout_") and name.endswith(".npz"):
                matches.append(os.path.join(dirpath, name))
    return matches


def _plot_selected_channels(logger: Logger, save_path: str, channels: list[str]) -> None:
    """Plot only the selected state channels to a single SVG."""
    # Use the first drone's counter to infer length; others are assumed same.
    n = int(logger.counters[0])
    if n <= 0:
        print(f"[WARN] Logger contains no samples, skipping plot: {save_path}")
        return
    t = np.arange(0, n / logger.LOGGING_FREQ_HZ, 1.0 / logger.LOGGING_FREQ_HZ)

    fig, axs = plt.subplots(len(channels), 1, figsize=(11, 3.0 * len(channels)), sharex=True)
    if len(channels) == 1:
        axs = [axs]

    for ax, name in zip(axs, channels):
        idx, ylabel = CHANNELS[name]
        for j in range(logger.NUM_DRONES):
            ax.plot(t, logger.states[j, idx, :n], label=f"drone_{j}")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.6)
        if logger.NUM_DRONES > 1:
            ax.legend(loc="upper right", fontsize=8, frameon=True)

    axs[-1].set_xlabel("time (s)")
    fig.tight_layout()

    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def convert_rollouts_to_svg(
    root: str,
    overwrite: bool = False,
    channels: list[str] | None = None,
    channels_raw: str = "",
) -> None:
    rollouts = _discover_rollouts(root)
    if not rollouts:
        print(f"[INFO] No kinematic_rollout_*.npz files found under {root!r}")
        return

    print(f"[INFO] Found {len(rollouts)} rollout archive(s) under {root!r}")

    for npz_path in rollouts:
        base, _ = os.path.splitext(npz_path)
        svg_path = f"{base}-{channels_raw}.svg"

        if not overwrite and os.path.exists(svg_path):
            print(f"[INFO] SVG already exists, skipping: {svg_path}")
            continue

        print(f"[INFO] Converting rollout to SVG: {npz_path}")
        logger = _load_logger_from_npz(npz_path)

        # Save only SVG here; PNGs are already produced by the original scripts.
        if channels:
            _plot_selected_channels(logger, svg_path, channels)
        else:
            logger.plot(save_path=[svg_path], show=False)
        print(f"[INFO] Saved kinematic rollout SVG: {svg_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SVG kinematic rollout plots from Logger-format "
            "kinematic_rollout_*.npz archives."
        )
    )
    parser.add_argument(
        "--root",
        default="results",
        type=str,
        help='Root directory to search (default: "results")',
        metavar="",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        type=lambda v: str(v).lower() in {"1", "true", "yes", "y"},
        help="Whether to overwrite existing SVG files (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--channels",
        default="all",
        type=str,
        help=(
            "Comma-separated state channels to plot (e.g. 'roll,pitch,yaw'). "
            "Valid names: "
            + ", ".join(sorted(CHANNELS.keys()))
            + ". Use 'all' to reproduce the full kinematic figure."
        ),
        metavar="",
    )

    args = parser.parse_args(argv)
    raw_channels = (args.channels or "").strip()
    channels_list: list[str] | None
    if raw_channels.lower() == "all" or not raw_channels:
        channels_list = None
    else:
        requested = [c.strip().lower() for c in raw_channels.split(",") if c.strip()]
        unknown = [c for c in requested if c not in CHANNELS]
        if unknown:
            print(
                "[WARN] Ignoring unknown channel name(s): "
                + ", ".join(unknown)
                + ". Valid names are: "
                + ", ".join(sorted(CHANNELS.keys()))
            )
        channels_list = [c for c in requested if c in CHANNELS]
        if not channels_list:
            print("[WARN] No valid channels requested, falling back to full kinematic plot.")
            channels_list = None

    convert_rollouts_to_svg(root=args.root, overwrite=args.overwrite, channels=channels_list, channels_raw=raw_channels)


if __name__ == "__main__":
    main()

