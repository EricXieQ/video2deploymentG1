# Copyright 2026 EricXieQ
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTICE OF CHANGES. This file is derived from policy/g1/asap_dab/export_wrapper.py
# in https://github.com/EricXieQ/rl_sar_g1_deployment (Apache-2.0), branch
# jetson-deploy, commit 314b2cb. Changes made for video2deploymentG1:
#   - --ckpt and --out are required; the hard-coded dab defaults are gone;
#   - the actor is loaded with strict key and shape checking;
#   - new --verify: compares the saved TorchScript policy with an independent,
#     loop-based re-implementation of the observation mapping and of the
#     checkpoint's MLP, on random inputs, and checks every weight tensor;
#   - --config optionally copies a policy config.yaml next to policy.pt, and like
#     policy.pt never replaces an existing one without --force;
#   - docstrings rewritten to be motion-agnostic.
#   The wrapper module (RlSarASAPWrapper) and its input/output layout are unchanged,
#   so a policy.pt exported here is interchangeable with one from the original script.
"""Export an ASAP motion-tracking checkpoint as an rl_sar TorchScript policy.pt.

ASAP's actor takes a 380-dim input, keys sorted alphabetically:
    [actions(23), base_ang_vel(3), dof_pos(23), dof_vel(23),
     history_actor(304), projected_gravity(3), ref_motion_phase(1)]
where history_actor holds 4 frames of each of those six terms, per key contiguous.

rl_sar (observations_history [0,1,2,3], priority "term") feeds a 376-dim vector:
    [actions_t..t-3 (4*29), ang_vel_t..t-3 (4*3), dof_pos_t..t-3 (4*29),
     dof_vel_t..t-3 (4*29), gravity_t..t-3 (4*3), phase_t..t-3 (4*1)]

The wrapper slices rl_sar's input into ASAP's layout, drops the 6 wrist joints
(29 -> 23), runs the actor and pads the 23 actions back to 29 (zeros on the wrists).

History convention: this is the layout shipped with the dab ("convention A":
history_actor gets rl_sar frames t..t-3; training used t-1..t-4). See
docs/TROUBLESHOOTING.md. It is kept so exported policies match the deployed dab.

Usage (hvgym env, from anywhere):
    python deploy/export_policy.py --ckpt <run>/model_N.pt \\
        --out <rl_sar>/policy/g1/asap_<name>/policy.pt --verify
"""


import argparse
import os
import shutil
import sys
from typing import Final

import torch
import torch.nn as nn

# Indices of the 23 ASAP-controlled joints inside the 29-DoF G1 (wrists 19-21, 26-28 dropped).
ASAP_DOF_INDICES = [0, 1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10, 11,
                    12, 13, 14,
                    15, 16, 17, 18,
                    22, 23, 24, 25]
assert len(ASAP_DOF_INDICES) == 23

RLSAR_OBS_DIM = 376
ASAP_OBS_DIM = 380


class ASAPActorModule(nn.Module):
    """Inner module: checkpoint keys are actor_module.module.{0,2,4,6}.{weight,bias}."""

    def __init__(self):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(380, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 23),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class ASAPActor(nn.Module):
    """Reconstruction of the ASAP MLP actor matching the checkpoint key structure."""

    def __init__(self):
        super().__init__()
        self.actor_module = ASAPActorModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor_module(x)


class RlSarASAPWrapper(nn.Module):
    """Bridge from rl_sar's term-priority history layout to ASAP's actor input."""

    HIST_LEN: Final[int] = 4
    NDOF: Final[int] = 29
    ASAP_NDOF: Final[int] = 23

    def __init__(self, asap_actor: ASAPActor):
        super().__init__()
        self.asap_actor = asap_actor
        self.register_buffer(
            "asap_dof_idx",
            torch.tensor(ASAP_DOF_INDICES, dtype=torch.long),
            persistent=False,
        )

    def forward(self, rl_sar_input: torch.Tensor) -> torch.Tensor:
        B = rl_sar_input.shape[0]
        H = self.HIST_LEN
        N = self.NDOF

        actions_4 = rl_sar_input[:, 0:116].reshape(B, H, N)
        ang_vel_4 = rl_sar_input[:, 116:128].reshape(B, H, 3)
        dof_pos_4 = rl_sar_input[:, 128:244].reshape(B, H, N)
        dof_vel_4 = rl_sar_input[:, 244:360].reshape(B, H, N)
        gravity_4 = rl_sar_input[:, 360:372].reshape(B, H, 3)
        phase_4 = rl_sar_input[:, 372:376].reshape(B, H, 1)

        actions_4_a = actions_4.index_select(2, self.asap_dof_idx)
        dof_pos_4_a = dof_pos_4.index_select(2, self.asap_dof_idx)
        dof_vel_4_a = dof_vel_4.index_select(2, self.asap_dof_idx)

        actions_t = actions_4_a[:, 0, :]
        ang_vel_t = ang_vel_4[:, 0, :]
        dof_pos_t = dof_pos_4_a[:, 0, :]
        dof_vel_t = dof_vel_4_a[:, 0, :]
        gravity_t = gravity_4[:, 0, :]
        phase_t = phase_4[:, 0, :]

        history_actor = torch.cat([
            actions_4_a.reshape(B, H * self.ASAP_NDOF),
            ang_vel_4.reshape(B, H * 3),
            dof_pos_4_a.reshape(B, H * self.ASAP_NDOF),
            dof_vel_4_a.reshape(B, H * self.ASAP_NDOF),
            gravity_4.reshape(B, H * 3),
            phase_4.reshape(B, H * 1),
        ], dim=-1)

        asap_input = torch.cat([
            actions_t, ang_vel_t, dof_pos_t, dof_vel_t,
            history_actor,
            gravity_t, phase_t,
        ], dim=-1)

        asap_action = self.asap_actor(asap_input)

        out = torch.zeros(B, self.NDOF, dtype=asap_action.dtype, device=asap_action.device)
        out.index_copy_(1, self.asap_dof_idx, asap_action)
        return out


def load_actor_state(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "actor_model_state_dict" not in ckpt:
        raise KeyError(f"{ckpt_path}: no actor_model_state_dict (keys: {sorted(ckpt)[:8]})")
    return {k: v for k, v in ckpt["actor_model_state_dict"].items() if k.startswith("actor_module.")}


def build_actor(actor_state: dict) -> ASAPActor:
    actor = ASAPActor()
    actor.load_state_dict(actor_state, strict=True)  # raises on missing, extra or mis-shaped keys
    actor.eval()
    return actor


# ---------------------------------------------------------------- verification

def reference_asap_obs(x: torch.Tensor) -> torch.Tensor:
    """Independent, loop-based construction of ASAP's 380-dim input from rl_sar's 376.

    Written from the layout description above rather than from the wrapper code.
    """
    terms = [("actions", 29, True), ("ang_vel", 3, False), ("dof_pos", 29, True),
             ("dof_vel", 29, True), ("gravity", 3, False), ("phase", 1, False)]
    rows = []
    for b in range(x.shape[0]):
        frames = {}  # term -> list of 4 frames (newest first), wrists dropped
        offset = 0
        for name, width, has_wrists in terms:
            fr = []
            for h in range(4):
                v = x[b, offset + h * width: offset + (h + 1) * width]
                if has_wrists:
                    v = torch.stack([v[i] for i in ASAP_DOF_INDICES])
                fr.append(v)
            frames[name] = fr
            offset += 4 * width
        assert offset == RLSAR_OBS_DIM
        # alphabetical keys: actions, base_ang_vel, dof_pos, dof_vel, history_actor,
        # projected_gravity, ref_motion_phase
        history = []
        for name, _, _ in terms:
            history.extend(frames[name])  # per key contiguous, 4 frames each
        row = [frames["actions"][0], frames["ang_vel"][0], frames["dof_pos"][0],
               frames["dof_vel"][0], torch.cat(history), frames["gravity"][0], frames["phase"][0]]
        rows.append(torch.cat(row))
    out = torch.stack(rows)
    assert out.shape[1] == ASAP_OBS_DIM
    return out


def reference_mlp(actor_state: dict, obs: torch.Tensor) -> torch.Tensor:
    """The checkpoint's MLP evaluated directly from its tensors (Linear + ELU, no ELU on the output)."""
    idx = sorted({int(k.split(".")[2]) for k in actor_state})
    h = obs
    for n, i in enumerate(idx):
        w = actor_state[f"actor_module.module.{i}.weight"]
        b = actor_state[f"actor_module.module.{i}.bias"]
        h = h @ w.T + b
        if n < len(idx) - 1:
            h = torch.nn.functional.elu(h)
    return h


def verify(policy_path: str, actor_state: dict, n: int, seed: int, tol: float) -> bool:
    torch.manual_seed(seed)
    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()

    ok = True
    # 1. weights: every actor tensor in the TorchScript file equals the checkpoint's
    sd = {k.replace("asap_actor.", "", 1): v for k, v in policy.state_dict().items()}
    for k, v in actor_state.items():
        if k not in sd or not torch.equal(sd[k], v):
            print(f"  FAIL weight {k} differs or is missing")
            ok = False
    print(f"  weights: {len(actor_state)} tensors compared, {'all equal' if ok else 'MISMATCH'}")

    # 2. outputs on random inputs (scaled like real observations, plus zeros)
    x = torch.randn(n, RLSAR_OBS_DIM) * 1.5
    x[0] = 0.0
    with torch.no_grad():
        got = policy(x).double()
        # reference in float64, so the comparison measures the mapping, not float32 rounding
        want23 = reference_mlp({k: v.double() for k, v in actor_state.items()},
                               reference_asap_obs(x).double())
    want = torch.zeros(n, 29, dtype=torch.float64)
    want[:, ASAP_DOF_INDICES] = want23
    diff = (got - want).abs().max().item()
    scale = max(1.0, want.abs().max().item())
    rel = diff / scale
    wrist = got[:, [19, 20, 21, 26, 27, 28]].abs().max().item()
    out_ok = got.shape == (n, 29) and rel <= tol and wrist == 0.0
    print(f"  outputs: {n} random inputs, shape {tuple(got.shape)}, max |policy.pt - checkpoint| = {diff:.3e} "
          f"on outputs up to {scale:.3g} (relative {rel:.2e}, tol {tol:g}); wrist outputs max {wrist:g} "
          f"-> {'PASS' if out_ok else 'FAIL'}")
    return ok and out_ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="ASAP checkpoint model_N.pt")
    p.add_argument("--out", required=True, help="output TorchScript policy.pt")
    p.add_argument("--config", help="optional config.yaml to copy next to policy.pt; "
                   "an existing config.yaml there is kept unless --force is given")
    p.add_argument("--verify", action="store_true", help="check policy.pt against the checkpoint")
    p.add_argument("--n", type=int, default=256, help="random inputs for --verify (default %(default)s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-5, help="max output difference relative to max(1, largest output) (default %(default)g)")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing --out and, with --config, an existing config.yaml beside it")
    args = p.parse_args(argv)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    config_dst = os.path.join(out_dir, "config.yaml")
    copy_config = bool(args.config) and os.path.abspath(args.config) != config_dst
    for path in [args.out] + ([config_dst] if copy_config else []):
        if os.path.exists(path) and not args.force:
            p.error(f"{path} exists; pass --force to overwrite")
    os.makedirs(out_dir, exist_ok=True)

    print(f"checkpoint {args.ckpt}")
    actor_state = load_actor_state(args.ckpt)
    actor = build_actor(actor_state)
    wrapper = RlSarASAPWrapper(actor).eval()
    scripted = torch.jit.script(wrapper)
    scripted.save(args.out)
    print(f"saved      {args.out}")

    if copy_config:
        shutil.copyfile(args.config, config_dst)
        print(f"config     {config_dst}")

    if args.verify:
        print("verify")
        if not verify(args.out, actor_state, args.n, args.seed, args.tol):
            print("VERIFY FAILED")
            return 1
        print("VERIFY PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
