"""Compare signed relative tracking error between two rollout ``kinematic_rollout_*.npz`` files.

Uses the same error definitions and time windows as ``rollout_rpy_error.py``. The first
path is file **A**, the second is **B**. Relative error uses **B** as the reference scale
when possible: ``(metric_B - metric_A) / denom``. **Positive** means **A** has strictly
smaller mean or max absolute error on that statistic (A is better).

Example
-------
From the repo root::

    $ python gym_pybullet_drones/examples/compare_error.py rollout_a.npz rollout_b.npz

"""
from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

from gym_pybullet_drones.examples.rollout_rpy_error import (
    _get_time_axis,
    attitude_errors,
    position_errors,
    summarize_errors,
    summarize_position_errors,
)


def _signed_relative(metric_a: float, metric_b: float) -> float:
    """``(metric_b - metric_a) / denom``; positive iff ``metric_a < metric_b``."""
    if metric_b != 0.0:
        return (metric_b - metric_a) / metric_b * 100
    if metric_a != 0.0:
        return (metric_b - metric_a) / metric_a * 100
    return 0.0


def _load_rollout(path: str) -> tuple[np.ndarray, np.ndarray, np.lib.npyio.NpzFile]:
    """Load ``states`` and ``controls`` after validating Logger layout."""
    try:
        data = np.load(path, allow_pickle=False)
    except OSError as e:
        raise ValueError(f"Could not load {path!r}: {e}") from e

    required = ("states", "controls")
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(
            f"Missing keys {missing} in {path!r}; have {list(data.files)}"
        )

    states = np.asarray(data["states"])
    controls = np.asarray(data["controls"])
    if states.ndim != 2 or states.shape[0] != 16:
        raise ValueError(f"Expected states with shape (16, n), got {states.shape}")
    if controls.ndim != 2 or controls.shape[0] != 12:
        raise ValueError(
            f"Expected controls with shape (12, n), got {controls.shape}"
        )
    if states.shape[1] != controls.shape[1]:
        raise ValueError(
            "states and controls must have same n; got "
            f"{states.shape[1]} vs {controls.shape[1]}"
        )
    return states, controls, data


def _summarize_window(
    e_x: np.ndarray,
    e_y: np.ndarray,
    e_z: np.ndarray,
    e_roll: np.ndarray,
    e_pitch: np.ndarray,
    e_yaw: np.ndarray,
    mask: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not np.any(mask):
        return None
    return (
        summarize_position_errors(e_x[mask], e_y[mask], e_z[mask]),
        summarize_errors(e_roll[mask], e_pitch[mask], e_yaw[mask]),
    )


def _print_block(
    label: str,
    pos_a: dict[str, Any],
    att_a: dict[str, Any],
    pos_b: dict[str, Any],
    att_b: dict[str, Any],
) -> None:
    print(label)
    for axis in ("roll", "pitch", "yaw"):
        sa, sb = att_a[axis], att_b[axis]
        mean_rel = _signed_relative(
            sa["mean_abs_error_rad"], sb["mean_abs_error_rad"]
        )
        max_rel = _signed_relative(sa["max_abs_error_rad"], sb["max_abs_error_rad"])
        print(
            f"  {axis:5s}  mean_rel = {mean_rel:+.4f}%   max_rel = {max_rel:+.4f}%"
        )
    for axis in ("x", "y", "z"):
        sa, sb = pos_a[axis], pos_b[axis]
        mean_rel = _signed_relative(
            sa["mean_abs_error_m"], sb["mean_abs_error_m"]
        )
        max_rel = _signed_relative(sa["max_abs_error_m"], sb["max_abs_error_m"])
        print(
            f"  {axis:5s}  mean_rel = {mean_rel:+.4f}%   max_rel = {max_rel:+.4f}%"
        )


def _print_block2(
        label: str,
        full_att_a: dict[str, Any],
        first_att_a: dict[str, Any],
        last_att_a: dict[str, Any],
        full_att_b: dict[str, Any],
        first_att_b: dict[str, Any],
        last_att_b: dict[str, Any],
        full_pos_a: dict[str, Any],
        first_pos_a: dict[str, Any],
        last_pos_a: dict[str, Any],
        full_pos_b: dict[str, Any],
        first_pos_b: dict[str, Any],
        last_pos_b: dict[str, Any],
) -> None:
    assert(label == "mean" or label == "max")
    print(label.capitalize()+":")
    rad = f"{label}_abs_error_rad"
    met = f"{label}_abs_error_m"
    print("[DoF], [Full 6s], [First 1s], [Last 1s],")
    for axis in ("roll",):
        full = _signed_relative(full_att_a[axis][rad], full_att_b[axis][rad])
        first = _signed_relative(first_att_a[axis][rad], first_att_b[axis][rad])
        last = _signed_relative(last_att_a[axis][rad], last_att_b[axis][rad])
        print(
            f"[{axis.capitalize()}], [{full:+.4g}%], [{first:+.4g}%], [{last:+.4g}%],"
        )

    for axis in ("x", "z"):
        full = _signed_relative(full_pos_a[axis][met], full_pos_b[axis][met])
        first = _signed_relative(first_pos_a[axis][met], first_pos_b[axis][met])
        last = _signed_relative(last_pos_a[axis][met], last_pos_b[axis][met])
        print(
            f"[{axis.capitalize()}], [{full:+.4g}%], [{first:+.4g}%], [{last:+.4g}%],"
        )
    print("")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Signed relative mean/max abs error between two rollout .npz files "
            "(Logger format). First file is A; positive means A has smaller error."
        )
    )
    parser.add_argument(
        "npz_a",
        type=str,
        help="Path to first rollout (A); positive relative error means A is better.",
    )
    parser.add_argument(
        "npz_b",
        type=str,
        help="Path to second rollout (B); used as reference scale in the denominator.",
    )
    parser.add_argument(
        "--typst_format",
        default=True,
        type=bool,
        help="Prints using a typst table format for copying",
    )
    args = parser.parse_args(argv)

    try:
        states_a, controls_a, data_a = _load_rollout(args.npz_a)
        states_b, controls_b, _data_b = _load_rollout(args.npz_b)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    n = states_a.shape[1]
    if states_b.shape[1] != n:
        print(
            f"[ERROR] Both rollouts must have the same n; got {n} vs "
            f"{states_b.shape[1]}",
            file=sys.stderr,
        )
        return 1

    try:
        t = _get_time_axis(data_a, n)
    except Exception as e:
        print(f"[ERROR] Could not determine time axis: {e}", file=sys.stderr)
        return 1

    first_mask = t <= 1.0
    last_mask = t >= (float(t[-1]) - 1.0)

    e_x_a, e_y_a, e_z_a = position_errors(states_a, controls_a)
    e_roll_a, e_pitch_a, e_yaw_a = attitude_errors(states_a, controls_a)
    e_x_b, e_y_b, e_z_b = position_errors(states_b, controls_b)
    e_roll_b, e_pitch_b, e_yaw_b = attitude_errors(states_b, controls_b)

    print(f"File A (first):  {args.npz_a}")
    print(f"File B (second): {args.npz_b}")
    print(f"Samples: {n}")
    print(f"Duration: {float(t[-1]):.6f} s")
    print(
        "Convention: mean_rel / max_rel compare mean |error| and max |error|; "
        "positive => first file (A) has smaller error on that statistic."
    )

    full_mask = np.ones(n, dtype=bool)

    if not args.typst_format:
        windows: list[tuple[str, np.ndarray]] = [
            ("Full:", full_mask),
            ("First 1s:", first_mask),
            ("Last 1s:", last_mask),
        ]

        for label, mask in windows:
            sums_a = _summarize_window(
                e_x_a, e_y_a, e_z_a, e_roll_a, e_pitch_a, e_yaw_a, mask
            )
            sums_b = _summarize_window(
                e_x_b, e_y_b, e_z_b, e_roll_b, e_pitch_b, e_yaw_b, mask
            )
            if sums_a is None or sums_b is None:
                print(f"{label} (no samples in window)")
                continue
            pos_a, att_a = sums_a
            pos_b, att_b = sums_b
            _print_block(label, pos_a, att_a, pos_b, att_b)
    else:
        def sumwin_a(_mask):
            return _summarize_window(
                e_x_a, e_y_a, e_z_a, e_roll_a, e_pitch_a, e_yaw_a, _mask
            )

        def sumwin_b(_mask):
            return _summarize_window(
                e_x_b, e_y_b, e_z_b, e_roll_b, e_pitch_b, e_yaw_b, _mask
            )

        full_pos_a, full_att_a = sumwin_a(full_mask)
        first_pos_a, first_att_a = sumwin_a(first_mask)
        last_pos_a, last_att_a = sumwin_a(last_mask)
        full_pos_b, full_att_b = sumwin_b(full_mask)
        first_pos_b, first_att_b = sumwin_b(first_mask)
        last_pos_b, last_att_b = sumwin_b(last_mask)

        _print_block2(
            "mean",
            full_att_a,
            first_att_a,
            last_att_a,
            full_att_b,
            first_att_b,
            last_att_b,
            full_pos_a,
            first_pos_a,
            last_pos_a,
            full_pos_b,
            first_pos_b,
            last_pos_b,
        )

        _print_block2(
            "max",
            full_att_a,
            first_att_a,
            last_att_a,
            full_att_b,
            first_att_b,
            last_att_b,
            full_pos_a,
            first_pos_a,
            last_pos_a,
            full_pos_b,
            first_pos_b,
            last_pos_b,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
