# Adding a new motion to rl_sar's G1 state machine

**Status.** The rl_sar change this page describes exists on a local rl_sar branch, not yet published (commit `b70d18d`, one commit on top of `jetson-deploy` 314b2cb). Until it lands on the public fork, `forks.lock` still pins 314b2cb, where the length is compiled in. The checklist at the end finishes this page once the commit is on the public fork.

## Why a code change was needed

On `jetson-deploy` 314b2cb the ASAP state `RLFSMStateRLASAPDab` in `src/rl_sar/fsm_robot/fsm_g1.hpp`:

- is entered from walking with key `5`;
- loads the policy folder named by the environment variable `RL_DAB_POLICY` (default `asap_dab`), so a new `policy/g1/asap_<name>/` folder can be chosen at run time without a rebuild;
- but sets **`rl.motion_length = 5.93f`**, the dab's duration, in code. rl_sar computes the phase as `ticks x 20 ms / motion_length` and returns to walking at phase 1. A clip of another length would get a phase clock running at the wrong speed, and would be cut off or overrun.

## Motion length (rl_sar with the change)

Each ASAP motion policy folder under `rl_sar/policy/g1/<name>/` declares its reference clip length in `config.yaml`:

```yaml
g1/asap_<name>:
  model_name: "policy.pt"
  motion_length: 6.27   # seconds; (frames - 1) / fps of the retargeted .pkl
```

- **Which number.** The length ASAP's motion library trained with: `(frames - 1) / 30` for a 30 fps pkl. `capture/check_reference.py` prints it on its "length" line. Dab: 179 frames, 5.93 s. Eric_moving_arms: 189 frames, 6.27 s. (Plain `frames / fps` would give 5.97 s for the dab, which is not what rl_sar ships.)
- **What it does.** rl_sar's ASAP state (key 5) uses it for the phase input the policy was trained on (`phase = t / motion_length`) and to decide when to hand back to the walking policy. If it is wrong, the policy sees the wrong phase: too short and the motion is cut off, too long and the phase runs slow.
- **Selecting the folder:** `RL_DAB_POLICY=asap_<name> ./run_g1.sh`, or in simulation `RL_DAB_POLICY=asap_<name> ./cmake_build/bin/rl_sim_mujoco g1 scene_29dof` (keys 0, R, 1, then 5). No rebuild needed.
- **Fallback.** A folder without the key, or with a value that is not a positive number (including the template's unedited `REPLACE_WITH_CLIP_SECONDS`), falls back to the dab's 5.93 s and prints `[DAB] ... has no positive motion_length; using 5.93 s` at entry. Check the `[DAB] motion_length = ... s` line in the console before trusting a run.
- **Scoring another motion:** `scripts/score_dab_session.py --motion-length <s> --win-raise <a> <b> --win-lean <a> <b>` (also `--dab-end-s <s>`, which overrides the abort line set by `--motion-length`). With no options it uses the dab's windows and gives byte-identical output to the 314b2cb script.
- **Checked on the branch:** the dab with the key at 5.93 still scored planted 10/10 with no falls in MuJoCo, a copy with `motion_length: 7.0` handed back at exactly 7.00 s of policy time, and both fallback cases printed the warning and ran normally.

`deploy/policy_template/config.yaml` carries the key as the non-numeric placeholder `REPLACE_WITH_CLIP_SECONDS`, with the `(frames - 1) / fps` formula and the dab's 5.93 as the example in its comment. A folder copied from the template and not edited therefore triggers the fallback warning instead of silently running on the dab's 5.93 s.

## To complete when the commit is on the public fork

- [ ] Bump the rl_sar line in `forks.lock` to the commit with the config-driven `motion_length`.
- [ ] Remove the "ignored on 314b2cb" note from `deploy/policy_template/config.yaml`.
- [ ] Run the MuJoCo gate (docs/PIPELINE.md step 10) on the rehearsal policy `asap_moving_arms` (docs/REHEARSAL_moving_arms.md) with `motion_length: 6.27` and scoring windows chosen for that clip.
