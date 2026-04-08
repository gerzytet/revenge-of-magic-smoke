"""Compute mean attitude error from a rollout ``kinematic_rollout_*.npz`` file.

The archives produced by ``MultiModelLearn.py`` and ``pid-vertical.py`` store Logger
layout: ``states`` is ``(16, n)`` with roll, pitch, yaw (rad) at indices 6--8, and
``controls`` is ``(12, n)`` with target roll, pitch, yaw at indices 3--5 (as in
``pid-vertical.py``). When controls are all zeros (e.g. RL rollouts), the reference
attitude is zero on each axis.

Example
-------
From the repo root::

    $ python gym_pybullet_drones/examples/rollout_rpy_error.py results/kinematic_rollout_cf2x.npz

"""
import argparse
import sys

import numpy as np


def _shortest_angle_diff(measured: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-sample shortest difference ``measured - reference`` in ``[-pi, pi]``."""
    return np.arctan2(np.sin(measured - reference), np.cos(measured - reference))


def _get_time_axis(data: np.lib.npyio.NpzFile, n: int) -> np.ndarray:
    """Return a 1D time axis aligned with samples."""
    if "t" in data.files:
        t = np.asarray(data["t"]).reshape(-1)
        if t.shape[0] != n:
            raise ValueError(f"t has length {t.shape[0]} but states has n={n}")
        return t
    if "timestamps" in data.files:
        ts = np.asarray(data["timestamps"]).reshape(-1)
        if ts.shape[0] == n:
            return ts
    if "logging_freq_hz" in data.files:
        freq = float(np.asarray(data["logging_freq_hz"]).reshape(()))
        if freq <= 0:
            raise ValueError(f"logging_freq_hz must be > 0, got {freq}")
        return np.arange(0, n / freq, 1.0 / freq)
    return np.arange(n, dtype=float)


def attitude_errors(
    states: np.ndarray,
    controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-time-step angular errors (rad) for roll, pitch, yaw."""
    r_meas = states[6, :]
    p_meas = states[7, :]
    y_meas = states[8, :]
    r_ref = controls[3, :]
    p_ref = controls[4, :]
    y_ref = controls[5, :]
    e_roll = _shortest_angle_diff(r_meas, r_ref)
    e_pitch = _shortest_angle_diff(p_meas, p_ref)
    e_yaw = _shortest_angle_diff(y_meas, y_ref)
    return e_roll, e_pitch, e_yaw


def summarize_errors(e_roll: np.ndarray, e_pitch: np.ndarray, e_yaw: np.ndarray) -> dict:
    """Mean absolute error and RMSE per axis (radians)."""
    names = ("roll", "pitch", "yaw")
    errs = (e_roll, e_pitch, e_yaw)
    out = {}
    for name, e in zip(names, errs):
        ae = np.abs(e)
        out[name] = {
            "mean_abs_error_rad": float(np.mean(ae)),
            "rmse_rad": float(np.sqrt(np.mean(e ** 2))),
            "n_samples": int(e.shape[0]),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mean roll / pitch / yaw error from a rollout .npz (Logger format)."
    )
    parser.add_argument(
        "npz_path",
        type=str,
        help="Path to kinematic_rollout_*.npz",
    )
    args = parser.parse_args(argv)

    try:
        data = np.load(args.npz_path, allow_pickle=False)
    except OSError as e:
        print(f"[ERROR] Could not load {args.npz_path!r}: {e}", file=sys.stderr)
        return 1

    required = ("states", "controls")
    missing = [k for k in required if k not in data.files]
    if missing:
        print(
            f"[ERROR] Missing keys {missing} in {args.npz_path!r}; have {list(data.files)}",
            file=sys.stderr,
        )
        return 1

    states = np.asarray(data["states"])
    controls = np.asarray(data["controls"])
    if states.ndim != 2 or states.shape[0] != 16:
        print(
            f"[ERROR] Expected states with shape (16, n), got {states.shape}",
            file=sys.stderr,
        )
        return 1
    if controls.ndim != 2 or controls.shape[0] != 12:
        print(
            f"[ERROR] Expected controls with shape (12, n), got {controls.shape}",
            file=sys.stderr,
        )
        return 1
    if states.shape[1] != controls.shape[1]:
        print(
            f"[ERROR] states and controls must have same n; got {states.shape[1]} vs {controls.shape[1]}",
            file=sys.stderr,
        )
        return 1

    n = states.shape[1]
    try:
        t = _get_time_axis(data, n)
    except Exception as e:
        print(f"[ERROR] Could not determine time axis: {e}", file=sys.stderr)
        return 1

    e_roll, e_pitch, e_yaw = attitude_errors(states, controls)
    full = summarize_errors(e_roll, e_pitch, e_yaw)

    first_mask = t <= 1.0
    last_mask = t >= (float(t[-1]) - 1.0)

    first = summarize_errors(e_roll[first_mask], e_pitch[first_mask], e_yaw[first_mask])
    last = summarize_errors(e_roll[last_mask], e_pitch[last_mask], e_yaw[last_mask])

    print(f"File: {args.npz_path}")
    print(f"Samples: {n}")
    print(f"Duration: {float(t[-1]):.6f} s")

    def _print_block(label: str, summary: dict) -> None:
        print(label)
        for axis in ("roll", "pitch", "yaw"):
            s = summary[axis]
            print(
                f"  {axis:5s}  mean |error| = {s['mean_abs_error_rad']:.6f} rad  "
                f"RMSE = {s['rmse_rad']:.6f} rad"
            )

    _print_block("Full:", full)
    _print_block("First 1s:", first)
    _print_block("Last 1s:", last)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
