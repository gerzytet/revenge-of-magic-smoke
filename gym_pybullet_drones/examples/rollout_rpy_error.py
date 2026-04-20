"""Compute mean position and attitude error from a rollout ``kinematic_rollout_*.npz`` file.

The archives produced by ``MultiModelLearn.py`` and ``pid-vertical.py`` store Logger
layout: ``states`` is ``(16, n)`` with position (m) at indices 0--2 and roll, pitch,
yaw (rad) at indices 6--8, and ``controls`` is ``(12, n)`` with target position at
indices 0--2 and target roll, pitch, yaw at indices 3--5 (as in ``pid-vertical.py``).
When controls are all zeros (e.g. RL rollouts), the reference pose is zero on each axis.

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
        return t
    if "timestamps" in data.files:
        ts = np.asarray(data["timestamps"]).reshape(-1)
        if ts.shape[0] == n:
            return ts
    if "logging_freq_hz" in data.files:
        freq = float(np.asarray(data["logging_freq_hz"]).reshape(()))
        return np.arange(0, n / freq, 1.0 / freq)
    return np.arange(n, dtype=float)


def position_errors(
    states: np.ndarray,
    controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-time-step position errors (m) for x, y, z (measured minus reference)."""
    x_meas, y_meas, z_meas = states[0, :], states[1, :], states[2, :]
    x_ref, y_ref, z_ref = controls[0, :], controls[1, :], controls[2, :]
    print("HACK ALERT: MultiModelLearn sets the control array to zero in the logger, even though the target position is currently [0,0,1]. z_ref is manually set to 1 here ")
    print("HACK ALERT: pid-vertical properly sets the control array, so z_ref is only set if z_ref is initially zero")

    if np.count_nonzero(z_ref) == 0:
        z_ref = np.ones_like(z_ref)
        
    return x_meas - x_ref, y_meas - y_ref, z_meas - z_ref


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


def summarize_position_errors(
    e_x: np.ndarray, e_y: np.ndarray, e_z: np.ndarray
) -> dict:
    """Mean / max absolute error and RMSE per axis (meters)."""
    names = ("x", "y", "z")
    errs = (e_x, e_y, e_z)
    out = {}
    for name, e in zip(names, errs):
        ae = np.abs(e)
        out[name] = {
            "mean_abs_error_m": float(np.mean(ae)),
            "max_abs_error_m": float(np.max(ae)),
            "rmse_m": float(np.sqrt(np.mean(e ** 2))),
            "n_samples": int(e.shape[0]),
        }
    return out


def summarize_errors(e_roll: np.ndarray, e_pitch: np.ndarray, e_yaw: np.ndarray) -> dict:
    """Mean / max absolute error and RMSE per axis (radians)."""
    names = ("roll", "pitch", "yaw")
    errs = (e_roll, e_pitch, e_yaw)
    out = {}
    for name, e in zip(names, errs):
        ae = np.abs(e)
        out[name] = {
            "mean_abs_error_rad": float(np.mean(ae)),
            "max_abs_error_rad": float(np.max(ae)),
            "rmse_rad": float(np.sqrt(np.mean(e ** 2))),
            "n_samples": int(e.shape[0]),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mean x/y/z and roll/pitch/yaw error from a rollout .npz (Logger format)."
    )
    parser.add_argument(
        "npz_path",
        type=str,
        help="Path to kinematic_rollout_*.npz",
    )
    parser.add_argument(
        "--typst_format",
        type=bool,
        default=True,
        help="Prints using a typst table format for copying",
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

    e_x, e_y, e_z = position_errors(states, controls)
    e_roll, e_pitch, e_yaw = attitude_errors(states, controls)
    full_pos = summarize_position_errors(e_x, e_y, e_z)
    full_att = summarize_errors(e_roll, e_pitch, e_yaw)

    first_mask = t <= 1.0
    last_mask = t >= (float(t[-1]) - 1.0)

    first_pos = summarize_position_errors(
        e_x[first_mask], e_y[first_mask], e_z[first_mask]
    )
    first_att = summarize_errors(
        e_roll[first_mask], e_pitch[first_mask], e_yaw[first_mask]
    )
    last_pos = summarize_position_errors(e_x[last_mask], e_y[last_mask], e_z[last_mask])
    last_att = summarize_errors(
        e_roll[last_mask], e_pitch[last_mask], e_yaw[last_mask]
    )

    print(f"File: {args.npz_path}")
    print(f"Samples: {n}")
    print(f"Duration: {float(t[-1]):.6f} s")

    def _print_block(label: str, pos_summary: dict, att_summary: dict) -> None:
        print(label)
        for axis in ("x", "y", "z"):
            s = pos_summary[axis]
            print(
                f"  {axis:5s}  mean |error| = {s['mean_abs_error_m']:.6f}   m  "
                f"max |error| = {s['max_abs_error_m']:.6f}   m  "
                #f"RMSE = {s['rmse_m']:.6f} m"
            )
        for axis in ("roll", "pitch", "yaw"):
            todeg = 180 / np.pi
            s = att_summary[axis]
            print(
                f"  {axis:5s}  mean |error| = {s['mean_abs_error_rad']*todeg:.6f} deg  "
                f"max |error| = {s['max_abs_error_rad']*todeg:.6f} deg  "
                #f"RMSE = {s['rmse_rad']:.6f} rad"
            )

    def _print_block_2(label: str, duration: float) -> None:
        assert(label == "mean" or label == "max")
        print(label.capitalize()+":")
        rad = f"{label}_abs_error_rad"
        met = f"{label}_abs_error_m"
        print(f"[DoF], [Full {duration:.2g}s], [First 1s], [Last 1s],")
        for axis in ("roll",):
            todeg = 180 / np.pi
            print(
                f"[{axis.capitalize()}], [{full_att[axis][rad]*todeg:.4g}#sym.degree], [{first_att[axis][rad]*todeg:.4g}#sym.degree], [{last_att[axis][rad]*todeg:.4g}#sym.degree],"
            )

        for axis in ("x", "z"):
            print(
                f"[{axis.capitalize()}], [{full_pos[axis][met]:.4g} m], [{first_pos[axis][met]:.4g} m], [{last_pos[axis][met]:.4g} m],"
            )
        print("")

    if not args.typst_format:
        _print_block("Full:", full_pos, full_att)
        _print_block("First 1s:", first_pos, first_att)
        _print_block("Last 1s:", last_pos, last_att)
    else:
        _print_block_2("mean", t[-1])
        _print_block_2("max", t[-1])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
