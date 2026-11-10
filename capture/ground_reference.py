#!/usr/bin/env python
"""Put the reference motion's feet on the floor, frame by frame.

TRAM's per-frame height is noisy and ASAP's retargeter grounds only one frame,
so retargeted references float: the dab's lower foot sat 4.5 to 9.7 cm above
the floor. A policy trained on that learns to track a floating ghost.

This script applies, to every frame t, a rigid vertical shift of the whole body:

    dz[t] = floor - min(z_left_ankle_roll_link[t], z_right_ankle_roll_link[t])
    root_trans_offset[t, 2] += dz[t]          (smpl_joints[t, :, 2] too, if present)

so the lower ankle_roll link sits exactly at `floor` (0.035 m by default, the
height of that link when the G1 stands flat-footed in Isaac Gym). Rotations,
joint angles, X, Y, timing and fps are untouched. This is the same dz the
grounded dab overlay used for display (data/asap-v6-dab-overlay-grounded),
applied to the data instead.

Limits: only the lower foot is pinned. During a real single-leg lift the
lifted foot keeps its height above the stance foot. A real knee bounce shows
up correctly as the pelvis dropping while the feet stay down.

The output is written beside the input with a `_grounded` suffix on both the
file name and the motion key. The input is never overwritten.

Example:
    python capture/ground_reference.py --asap-root $ASAP_ROOT \\
        $ASAP_ROOT/humanoidverse/data/motions/.../0-motions_raw_tairantestbed_smpl_<name>_amass.pkl
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_fk as rf  # noqa: E402


def ground(motion: dict, robot: rf.RobotModel, floor: float) -> tuple[dict, np.ndarray]:
    feet = rf.body_heights(robot, motion)
    dz = floor - feet.min(axis=1)
    out = dict(motion)
    trans = np.array(motion["root_trans_offset"], copy=True)
    trans[:, 2] = trans[:, 2] + dz.astype(trans.dtype)
    out["root_trans_offset"] = trans
    if "smpl_joints" in motion:
        sj = np.array(motion["smpl_joints"], copy=True)
        sj[:, :, 2] = sj[:, :, 2] + dz[:, None].astype(sj.dtype)
        out["smpl_joints"] = sj
    return out, dz


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkl", help="retargeted motion pkl (one motion)")
    ap.add_argument("--asap-root", default=rf.default_asap_root(),
                    help="ASAP checkout, used to find the robot MJCF (default: $ASAP_ROOT)")
    ap.add_argument("--mjcf", help=f"robot MJCF (default: <asap-root>/{rf.DEFAULT_MJCF})")
    ap.add_argument("--floor", type=float, default=rf.FLOOR_ANKLE_ROLL_Z,
                    help="target height of the lower ankle_roll link, m (default %(default)s)")
    ap.add_argument("--suffix", default="_grounded", help="added to file name and motion key")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = ap.parse_args(argv)

    mjcf = args.mjcf or (args.asap_root and os.path.join(args.asap_root, rf.DEFAULT_MJCF))
    if not mjcf:
        ap.error("pass --asap-root or --mjcf (or set ASAP_ROOT)")

    name, motion = rf.load_motion(args.pkl)
    stem, ext = os.path.splitext(args.pkl)
    out_path = f"{stem}{args.suffix}{ext}"
    if os.path.abspath(out_path) == os.path.abspath(args.pkl):
        ap.error("output would overwrite the input; use a non-empty --suffix")
    if os.path.exists(out_path) and not args.force:
        ap.error(f"{out_path} exists; pass --force to replace it")

    robot = rf.load_robot(mjcf)
    grounded, dz = ground(motion, robot, args.floor)
    feet_after = rf.body_heights(robot, grounded).min(axis=1)

    out_name = name + args.suffix
    # ASAP's motion library derives the key from the file name, so the key must
    # match the new file name (see motion_lib_base.load_motion_with_skeleton).
    expected_key = os.path.basename(out_path).split(".")[0]
    if out_name != expected_key:
        print(f"note: motion key '{out_name}' differs from file stem '{expected_key}'; using the file stem")
        out_name = expected_key
    joblib.dump({out_name: grounded}, out_path)

    fps = motion["fps"]
    step = np.abs(np.diff(dz)).max() if len(dz) > 1 else 0.0
    print(f"input : {args.pkl}")
    print(f"output: {out_path}  (key {out_name})")
    print(f"frames {len(dz)} at {fps} fps; floor {args.floor:.4f} m")
    print(f"dz     min {dz.min()*100:+.2f} cm  max {dz.max()*100:+.2f} cm  mean {dz.mean()*100:+.2f} cm  "
          f"largest frame-to-frame change {step*1000:.1f} mm")
    print(f"lower ankle_roll z after: min {feet_after.min():.4f}  max {feet_after.max():.4f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
