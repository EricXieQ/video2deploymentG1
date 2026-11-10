#!/usr/bin/env python
"""Sanity-check a retargeted reference motion before spending GPU hours on it.

Prints one line per check, each marked PASS, WARN or INFO. Thresholds come from
the runbook (docs/PIPELINE.md, step 5) and from what went wrong with the dab:

  fps                30 (the retargeter always writes 30; anything else is a bug)
  length             4 to 7 s (the dab was 5.93 s)
  lowest-foot height per frame: its range must be within 1 cm (the dab varied 5 cm),
                     and it must stay within 1 cm of the floor constant (0.035 m)
  root height range  INFO only: a real bounce moves the pelvis; floating shows up
                     in the foot line above
  pelvis tilt        mean under 15 deg (the dab sat at 22 to 25 deg; the paper's
                     clips mostly 4 to 15); also printed for the first and last 1 s
  joint speed        peak under 0.9 x the joint's velocity limit (training soft limit)
  joint limits       no joint within 0.02 rad of a limit for 0.2 s or longer
  start/end pose     largest joint difference from the walking stance under 0.5 rad
                     (a heuristic: the policy is trained to enter from that stance)

Exit status is 0 even with warnings; it is 2 only if the file cannot be read.
Use --strict to exit 1 when any WARN is printed.

Example:
    python capture/check_reference.py --asap-root $ASAP_ROOT path/to/motion.pkl
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_fk as rf  # noqa: E402

FOOT_RANGE_MAX = 0.01      # m
FOOT_FLOOR_TOL = 0.01      # m
TILT_MAX_DEG = 15.0
SPEED_FRACTION = 0.9
LIMIT_MARGIN = 0.02        # rad
LIMIT_MAX_S = 0.2
STANCE_MAX_RAD = 0.5
LENGTH_RANGE_S = (4.0, 7.0)


class Report:
    def __init__(self):
        self.warnings = 0

    def line(self, status: str, what: str, detail: str):
        if status == "WARN":
            self.warnings += 1
        print(f"[{status:4s}] {what:<22s} {detail}")

    def check(self, ok: bool, what: str, detail: str):
        self.line("PASS" if ok else "WARN", what, detail)


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkl", help="retargeted motion pkl (one motion)")
    ap.add_argument("--asap-root", default=rf.default_asap_root(),
                    help="ASAP checkout, for the robot MJCF and joint limits (default: $ASAP_ROOT)")
    ap.add_argument("--mjcf", help=f"robot MJCF (default: <asap-root>/{rf.DEFAULT_MJCF})")
    ap.add_argument("--robot-yaml", help=f"robot config with joint limits (default: <asap-root>/{rf.DEFAULT_ROBOT_YAML})")
    ap.add_argument("--floor", type=float, default=rf.FLOOR_ANKLE_ROLL_Z,
                    help="flat-foot height of the ankle_roll link, m (default %(default)s)")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any check warns")
    args = ap.parse_args(argv)

    mjcf = args.mjcf or (args.asap_root and os.path.join(args.asap_root, rf.DEFAULT_MJCF))
    robot_yaml = args.robot_yaml or (args.asap_root and os.path.join(args.asap_root, rf.DEFAULT_ROBOT_YAML))
    if not mjcf or not robot_yaml:
        ap.error("pass --asap-root (or set ASAP_ROOT), or both --mjcf and --robot-yaml")

    try:
        name, motion = rf.load_motion(args.pkl)
    except Exception as e:  # noqa: BLE001
        print(f"cannot read {args.pkl}: {e}", file=sys.stderr)
        return 2

    robot = rf.load_robot(mjcf)
    dof_names, lo, hi, vmax = rf.load_joint_limits(robot_yaml)
    if [n for n in robot.joint_names] != list(dof_names):
        print("note: MJCF joint order differs from robot yaml dof_names; limits are matched by name")
        order = [list(dof_names).index(n) for n in robot.joint_names]
        lo, hi, vmax = lo[order], hi[order], vmax[order]
        dof_names = robot.joint_names

    fps = int(motion["fps"])
    dof = np.asarray(motion["dof"], dtype=np.float64)
    trans = np.asarray(motion["root_trans_offset"], dtype=np.float64)
    T = dof.shape[0]
    length = (T - 1) / fps  # what ASAP's motion library uses as the clip duration
    rep = Report()

    print(f"motion {name}")
    print(f"file   {args.pkl}")
    rep.check(fps == 30, "fps", f"{fps}")
    rep.check(LENGTH_RANGE_S[0] <= length <= LENGTH_RANGE_S[1], "length",
              f"{T} frames = {length:.2f} s of motion (aim {LENGTH_RANGE_S[0]:.0f} to {LENGTH_RANGE_S[1]:.0f} s)")

    feet = rf.body_heights(robot, motion)
    lowest = feet.min(axis=1)
    frange = lowest.max() - lowest.min()
    rep.check(frange <= FOOT_RANGE_MAX, "lowest-foot range",
              f"{frange*100:.1f} cm (z {lowest.min():.3f} to {lowest.max():.3f} m; pass <= {FOOT_RANGE_MAX*100:.0f} cm)")
    off = np.abs(lowest - args.floor).max()
    rep.check(off <= FOOT_FLOOR_TOL, "foot on floor",
              f"worst frame {off*100:.1f} cm from floor {args.floor:.3f} m "
              f"(mean lowest foot {lowest.mean():.3f} m; pass <= {FOOT_FLOOR_TOL*100:.0f} cm)")
    rep.line("INFO", "root height", f"{trans[:,2].min():.3f} to {trans[:,2].max():.3f} m "
             f"(range {np.ptp(trans[:,2])*100:.1f} cm)")

    tilt = rf.pelvis_tilt_deg(motion)
    n1 = min(fps, T)
    rep.check(tilt.mean() < TILT_MAX_DEG, "pelvis tilt",
              f"mean {tilt.mean():.1f} deg, range {tilt.min():.1f} to {tilt.max():.1f}; "
              f"first 1 s {tilt[:n1].mean():.1f}, last 1 s {tilt[-n1:].mean():.1f} (pass mean < {TILT_MAX_DEG:.0f})")

    vel = np.abs(np.diff(dof, axis=0)) * fps if T > 1 else np.zeros((1, dof.shape[1]))
    peak = vel.max(axis=0)
    ratio = peak / vmax
    j = int(np.argmax(ratio))
    arm = [i for i, n in enumerate(dof_names) if "shoulder" in n or "elbow" in n]
    rep.check(ratio.max() < SPEED_FRACTION, "peak joint speed",
              f"{peak[j]:.1f} rad/s at {dof_names[j]} = {ratio[j]*100:.0f}% of {vmax[j]:.0f} rad/s; "
              f"arms peak {peak[arm].max():.1f} rad/s (pass < {SPEED_FRACTION*100:.0f}% of limit)")

    near = (dof <= lo + LIMIT_MARGIN) | (dof >= hi - LIMIT_MARGIN)
    runs = np.array([longest_run(near[:, i]) for i in range(dof.shape[1])])
    worst = int(np.argmax(runs))
    if runs.max() == 0:
        rep.check(True, "joint limits", "no joint within 0.02 rad of a limit")
    else:
        rep.check(runs.max() / fps < LIMIT_MAX_S, "joint limits",
                  f"longest stay at a limit {runs.max()/fps:.2f} s at {dof_names[worst]} "
                  f"(pass < {LIMIT_MAX_S} s)")

    for label, idx in (("start pose", 0), ("end pose", -1)):
        d = np.abs(dof[idx] - rf.LOCO_STANCE_23DOF)
        k = int(np.argmax(d))
        rep.check(d.max() < STANCE_MAX_RAD, label,
                  f"largest difference from walking stance {d.max():.2f} rad at {dof_names[k]}, "
                  f"mean {d.mean():.2f} rad; pelvis z {trans[idx,2]:.3f} m vs 0.78 at entry "
                  f"(pass max < {STANCE_MAX_RAD} rad)")

    print(f"{rep.warnings} warning(s)")
    return 1 if (args.strict and rep.warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
