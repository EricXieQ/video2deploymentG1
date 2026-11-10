"""Shared helpers for check_reference.py and ground_reference.py.

A retargeted ASAP reference motion is a joblib pickle holding one dict
{motion_name: {...}} with these fields (written by ASAP's
scripts/data_process/fit_smpl_motion.py):

    root_trans_offset  (T, 3)      pelvis position, metres, world frame, Z up
    root_rot           (T, 4)      pelvis orientation, quaternion x, y, z, w
    dof                (T, 23)     joint angles, radians, in the robot's MJCF order
    pose_aa            (T, 27, 3)  the same pose as axis-angle per body (used by training FK)
    smpl_joints        (T, 24, 3)  SMPL joint positions (not used by training)
    fps                int         always 30 after retargeting

Forward kinematics here use MuJoCo on the same MJCF file ASAP's motion library
parses (robot.motion.asset in config/robot/g1/g1_29dof_anneal_23dof.yaml), so
link positions match what the training environment sees as the reference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np

# Relative to the ASAP checkout root.
DEFAULT_MJCF = "humanoidverse/data/robots/g1/g1_29dof_anneal_23dof_fitmotionONLY.xml"
DEFAULT_ROBOT_YAML = "humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml"

# Median ankle_roll link height of the trained v6 dab policy standing with flat
# feet in Isaac Gym (data/asap-v6-dab-overlay-grounded, FLOOR = 0.035002 m).
# The link origin sits this far above the sole, so "foot on the floor" means
# the lower ankle_roll link is at this height, not at zero.
FLOOR_ANKLE_ROLL_Z = 0.035

FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")

# The walking policy's default stance in ASAP's 23-DoF order. Copied from
# humanoidverse/envs/motion_tracking/motion_tracking.py (LOCO_STANCE_ASAP_23DOF)
# in the ASAP fork at tag dab-v6-recipe. The off-trajectory entry trains the
# policy to start from this pose; a reference that starts close to it makes the
# hand-off from walking easier.
LOCO_STANCE_23DOF = np.array([
    -0.20, 0.00, 0.00, 0.42, -0.23, 0.00,
    -0.20, 0.00, 0.00, 0.42, -0.23, 0.00,
    0.00, 0.00, 0.00,
    0.35, 0.18, 0.00, 0.87,
    0.35, -0.18, 0.00, 0.87,
])


def default_asap_root() -> str | None:
    return os.environ.get("ASAP_ROOT")


def load_motion(path: str) -> tuple[str, dict]:
    data = joblib.load(path)
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"{path}: expected a dict with exactly one motion, got {type(data)} "
                         f"with {len(data) if isinstance(data, dict) else '?'} entries")
    (name, motion), = data.items()
    for key in ("root_trans_offset", "root_rot", "dof", "fps"):
        if key not in motion:
            raise ValueError(f"{path}: motion '{name}' has no '{key}' field")
    return name, motion


@dataclass
class RobotModel:
    model: object
    data: object
    joint_names: list
    body_ids: dict


def load_robot(mjcf_path: str) -> RobotModel:
    import mujoco  # imported here so --help works without MuJoCo

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"{mjcf_path}: first joint is not a free joint")
    joint_names = [model.joint(i).name for i in range(1, model.njnt)]
    body_ids = {name: model.body(name).id for name in FOOT_LINKS + ("pelvis",)}
    return RobotModel(model, data, joint_names, body_ids)


def body_heights(robot: RobotModel, motion: dict, bodies=FOOT_LINKS) -> np.ndarray:
    """World Z of each named body for every frame, shape (T, len(bodies))."""
    import mujoco

    trans = np.asarray(motion["root_trans_offset"], dtype=np.float64)
    quat_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
    dof = np.asarray(motion["dof"], dtype=np.float64)
    if dof.shape[1] != robot.model.nq - 7:
        raise ValueError(f"motion has {dof.shape[1]} joints, MJCF has {robot.model.nq - 7}")
    out = np.zeros((trans.shape[0], len(bodies)))
    ids = [robot.model.body(b).id for b in bodies]
    for t in range(trans.shape[0]):
        robot.data.qpos[0:3] = trans[t]
        robot.data.qpos[3:7] = quat_xyzw[t, [3, 0, 1, 2]]  # MuJoCo wants w, x, y, z
        robot.data.qpos[7:] = dof[t]
        mujoco.mj_kinematics(robot.model, robot.data)
        out[t] = robot.data.xpos[ids, 2]
    return out


def pelvis_tilt_deg(motion: dict) -> np.ndarray:
    """Angle between the pelvis Z axis and world Z, degrees, per frame."""
    from scipy.spatial.transform import Rotation as R

    zaxis = R.from_quat(np.asarray(motion["root_rot"], dtype=np.float64)).as_matrix()[:, :, 2]
    return np.degrees(np.arccos(np.clip(zaxis[:, 2], -1.0, 1.0)))


def load_joint_limits(robot_yaml: str):
    import yaml

    with open(robot_yaml) as f:
        cfg = yaml.safe_load(f)["robot"]
    return (cfg["dof_names"],
            np.array(cfg["dof_pos_lower_limit_list"]),
            np.array(cfg["dof_pos_upper_limit_list"]),
            np.array(cfg["dof_vel_limit_list"]))
