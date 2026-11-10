# Adding a new motion to rl_sar's G1 state machine

**Status: waiting on an rl_sar change being made in parallel.** This page will be completed when that change lands on the rl_sar fork and `forks.lock` is bumped to it.

## Why a code change is needed

On the pinned rl_sar commit (`jetson-deploy` 314b2cb), the dab state `RLFSMStateRLASAPDab` in `src/rl_sar/fsm_robot/fsm_g1.hpp`:

- is entered from walking with key `5`;
- loads the policy folder named by the environment variable `RL_DAB_POLICY` (default `asap_dab`), so a new `policy/g1/asap_<name>/` folder can already be selected at runtime without a rebuild;
- but sets **`rl.motion_length = 5.93f`**, the dab's duration, in code. rl_sar computes the phase as `ticks x 20 ms / motion_length` (`rl_sdk.cpp`) and returns to walking at phase 1. A motion of another length would get a phase clock running at the wrong speed and be cut off or overrun.

## The planned change

Make `motion_length` a key in the policy's `config.yaml`, read by the state on entry, with 5.93 as the default so `asap_dab` behaves exactly as before. Then any motion runs through the same state:

```bash
RL_DAB_POLICY=asap_my_motion ./cmake_build/bin/rl_sim_mujoco g1 scene_29dof    # keys 0, R, 1, then 5
```

The value to put in `config.yaml` is the clip's duration as ASAP's motion library computes it, `(frames - 1) / 30` seconds; `capture/check_reference.py` prints it on its "length" line (Eric_moving_arms: 189 frames = 6.27 s).

(The alternative considered was a separate state on a free key, `6`, `7` or `8`.)

## To complete when the change lands

- [ ] Bump the rl_sar line in `forks.lock` to the commit with config-driven `motion_length`.
- [ ] Add the new key to `deploy/policy_template/config.yaml` with a "MUST MATCH the clip" comment.
- [ ] Document the exact key name, default, and the log line that confirms the value at entry.
- [ ] Run the MuJoCo gate (docs/PIPELINE.md step 10) on the rehearsal policy `asap_moving_arms` (docs/REHEARSAL_moving_arms.md) with scoring windows for that clip.
- [ ] Note the score windows in rl_sar's `scripts/score_dab_session.py` that need arguments (`WIN_RAISE`, `WIN_LEAN`, `DAB_END_S`).
