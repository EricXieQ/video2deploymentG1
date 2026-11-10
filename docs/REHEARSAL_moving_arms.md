# Rehearsal: steps 5 to 8 on `Eric_moving_arms`

**Purpose.** Before recording the 6-7 gesture, run the new parts of the pipeline end to end on a clip that was already through TRAM and retargeting: the reference check and grounding, the two-stage training recipe, and the export. And prove that the public recipe runs from the public ASAP fork, with nothing from a private checkout except the inputs listed in `docs/ASSETS.md`.

**What it can and cannot show.** `Eric_moving_arms` is slow arm waving (dominant frequencies 0.16 to 0.8 Hz, peak arm joint speed 3.9 rad/s), about ten times slower than a 6-7 gesture. The rehearsal tests the plumbing and whether the compressed stage 2 trains stably. It says nothing about the 6-7's speed or torque demands. The training budgets (3,000 + 1,500 iterations) are short tests, not a deployable policy.

Date: 2026-11-09. Machine: one RTX 4090 (24 GB), Ubuntu 22.04, conda env `hvgym` (Python 3.8, torch 2.4.1, Isaac Gym Preview 4).

## Environment: the public fork, nothing else

```bash
git clone --branch dab-recipe-v6 https://github.com/EricXieQ/ASAP.git asap_public    # HEAD af60acf = tag dab-v6-recipe
export ASAP_ROOT=$PWD/asap_public PYTHONPATH=$PWD/asap_public
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/hvgym/lib CONDA_ENV_PREFIX=$HOME/miniconda3/envs/hvgym
python -c "import humanoidverse; print(humanoidverse.__file__)"    # -> asap_public/humanoidverse/__init__.py
```
The inputs the fork does not ship (the G1 shape fit, the Eric_moving_arms npz and pkl) were copied into the clone at their usual paths; hashes in `docs/ASSETS.md`.

## Result in one paragraph

The public recipe runs from the public fork with no code changes. Steps 5 to 8 all completed. Grounding brought the reference's lower foot from a 9.7 cm range to exactly the floor constant on all 189 frames. Stage 1 (3,000 iterations, 63 min) trained with the same curve shape as the dab's v1. **Stage 2, the compressed single jump to the deployment flag set, trained stably**: no NaN, no NaN-guard hit, and episode length dipped in the first 170 iterations, then recovered past its stage-1 level and was still rising when the 1,500-iteration budget ran out. But **the resulting policy is not good yet**: in Isaac Gym it fell in 4 of 10 episodes without pushes, late in the clip. That is expected from 1.5 GPU-hours of training, against the dab's about 70 GPU-hours to its deployed checkpoint. The export verified bit-exact. The public fork was missing nothing needed to train. The only missing inputs were the motion files and shape fit, which are user data.

## Step 5. Check and ground the reference

```bash
PKL=$ASAP_ROOT/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_Eric_moving_arms_amass.pkl
python capture/check_reference.py  $PKL                   # 0.3 s
python capture/ground_reference.py $PKL                   # 0.3 s, writes ..._amass_grounded.pkl
python capture/check_reference.py  ${PKL%.pkl}_grounded.pkl
```

| Check | Before | After grounding |
|---|---|---|
| fps, length | PASS 30, 189 frames = 6.27 s | same |
| lowest-foot range | **WARN 9.7 cm** (0.045 to 0.142 m) | **PASS 0.0 cm** (0.0350 m on every frame) |
| foot on floor | WARN, worst frame 10.7 cm above 0.035 m | PASS, 0.0 cm |
| root height (info) | 0.798 to 0.892 m, range **9.4 cm** | 0.785 to 0.791 m, range 0.6 cm |
| pelvis tilt | **WARN mean 22.5 deg** (20.7 to 25.8) | same (grounding does not rotate) |
| peak joint speed | PASS 3.9 rad/s, left shoulder pitch, 11 % of limit | same |
| joint limits | WARN waist pitch at its 0.52 rad limit for 1.03 s | same |
| start / end pose vs walking stance | WARN 0.52 rad (waist pitch) / PASS 0.49 rad | same |

The grounding shift `dz` ranged from -10.7 to -1.0 cm (mean -5.6 cm; the reference was moved down on every frame). Its largest frame-to-frame change was 11.7 mm. Both expected warnings appeared (floating root 9.4 cm, tilt 22.5 deg), and the foot is constant within 1 cm after grounding. The tilt and the waist pitch parked at its limit are what remains: they come from retargeting, not from the floor. The same FK gives the dab pkl's known numbers (lower ankle_roll link at 4.5 to 9.7 cm, i.e. 1 to 6 cm above its flat-foot height; root 0.800 to 0.851 m), which cross-checks `check_reference.py` against the earlier overlay analysis.

## Step 6. Stage 1 from scratch

```bash
MOTION_PKL=${PKL%.pkl}_grounded.pkl PROJECT=MovingArmsRehearsal EXPERIMENT=MovingArms_stage1 \
ITERATIONS=3000 NUM_ENVS=4096 bash train/stage1_pretrain.sh
```
- **First attempt failed after 3 s:** `RuntimeError: Ninja is required to load C++ extensions`. Calling the env's python by full path without `conda activate` leaves `ninja` off `PATH`, and gymtorch compiles on first import. Fixed in the script (it now puts `$CONDA_ENV_PREFIX/bin` on `PATH`) and relaunched. Not a fork problem.
- **Run:** 2026-11-09 23:15 to 00:18, 3,760 s wall (62.7 min, 1.04 h of training at about 2,900 iterations per hour). 4096 robots, headless. 297 MB of checkpoints.
- **Config check:** the saved `config.yaml` differs from the dab's v1 `config.yaml` only in project and experiment name, `num_learning_iterations`, `motion_file`, `ctrl_delay_step_range` `[0, 2]` to `[0, 3]`, and the dab's stray `visualize_motion_file: null`.

TensorBoard, mean over the 50 iterations up to each mark:

| iteration | 499 | 999 | 1499 | 1999 | 2499 | 2999 |
|---|---|---|---|---|---|---|
| `Train/mean_episode_length` (of 313) | 83.6 | **120.1** | 103.0 | 76.1 | 62.0 | 61.4 |
| motion-far termination threshold, m | 1.19 | 0.88 | 0.65 | 0.48 | 0.36 | 0.30 |
| upper-body tracking error | 0.350 | 0.278 | 0.235 | 0.191 | 0.150 | 0.129 |
| lower-body tracking error | 0.208 | 0.194 | 0.163 | 0.143 | 0.116 | 0.099 |
| `rew_teleop_body_position_extend` | 0.068 | 0.104 | 0.102 | 0.085 | 0.077 | 0.078 |
| dab v1, episode length (of 297) | 77 | 108 | 107 | 99 | 84 | 81 |
| dab v1, upper / lower error | 0.36 / 0.21 | 0.26 / 0.17 | 0.21 / 0.14 | 0.17 / 0.12 | 0.13 / 0.10 | 0.12 / 0.08 |

The episode length falls after iteration 1000 because the termination curriculum tightens the "too far from the reference" threshold from 1.5 m to 0.3 m. The per-step tracking error keeps falling throughout. The dab's v1 shows the same dip (108 to 81) and recovered to about 120 by iteration 10,000. The rehearsal's errors at 3,000 are close to the dab's. Non-finite values: none.

**Headless evaluation** (Isaac Gym, 1 robot, no pushes, seed 0; the author's camera-recorder tool, not published, run unchanged): 10 of 10 episodes ran the full clip without a fall. But the robot tracks **crouched**: minimum base height about 0.63 m, against about 0.79 m in the reference. Mean body-position error 0.138 m, joint error 0.42 rad. With pushes (3 pushes over 6 episodes), 4 of 6 episodes ended early.

Stills (in the hand-off folder's `eval/`, not published): `stage1_model3000_t{00.50,02.00,03.50,05.00}s.png`, contact sheet `stage1_contact.png`; curves `stage1_curves.png`.

## Step 7. Stage 2, deployment fine-tune

```bash
MOTION_PKL=${PKL%.pkl}_grounded.pkl PROJECT=MovingArmsRehearsal EXPERIMENT=MovingArms_stage2 \
CHECKPOINT=<stage-1 run>/model_3000.pt ITERATIONS=1500 bash train/stage2_deploy_finetune.sh
```
- **Run:** 2026-11-10 00:18 to 00:43, 1,484 s wall (24.7 min, about 3,700 iterations per hour), iterations 3000 to 4500. The optimizer state was loaded from the checkpoint, as in the dab's fine-tunes.
- **Config check:** the saved `config.yaml` differs from the dab's v6 `config.yaml` only in checkpoint, names, `num_learning_iterations`, `motion_file`, the delay `[0, 3]` and `visualize_motion_file`. So the single stage really is v6's flag set.

| iteration | 3000 (first) | 3249 | 3499 | 3749 | 3999 | 4249 | 4499 |
|---|---|---|---|---|---|---|---|
| `Train/mean_episode_length` (of 313) | 19.8 | 41.5 | 76.3 | 101.3 | 114.0 | 120.5 | **124.6** |
| `Train/mean_reward` | -20.5 | -3.0 | -0.70 | -0.25 | 0.20 | 0.63 | 0.79 |
| joint position error (norm) | 3.08 | 2.37 | 1.95 | 1.75 | 1.62 | 1.60 | 1.60 |
| upper / lower-body tracking error | 0.18 / 0.09 | 0.24 / 0.11 | 0.24 / 0.13 | 0.24 / 0.14 | 0.23 / 0.13 | 0.22 / 0.13 | 0.22 / 0.13 |
| policy noise std | 0.79 | 0.36 | 0.31 | 0.33 | 0.35 | 0.35 | 0.35 |

**Stability verdict: stable.** No non-finite value in any TensorBoard scalar, zero `[NaN GUARD]` lines in the log, and the episode length did not collapse. It fell from about 90 to a minimum of about 23 around iteration 3170, then rose steadily to 125, above stage 1's best. The early dip resembles the dab's own v3 jump, and likely has the same causes (new tilt and low-height terminations, the reward change, and 20 % of episodes starting from the walking stance): v3 started at a mean reward of -193 and dipped to an episode length of 54 before reaching 141. The curve was still rising at 4,500, so the budget, not the recipe, ended it.

**Headless evaluation of `model_4500`** (same tool and settings): the robot now stands at about the reference height (minimum base height 0.73 to 0.75 m in completed episodes) and the joint error halved (0.23 rad against 0.42). But **4 of 10 episodes without pushes fell**, at 4.5 to 5.3 s into the 6.27 s clip (base height 0.53 m when terminated). With pushes, 5 of 6 ended early. For comparison, the dab's v6 holds a training episode length of 178 of 179 possible.

Stills: `stage2_model4500_t{...}s.png`, `stage2_contact.png`; curves `stage2_curves.png`.

What this means for the 6-7: the two-stage recipe is usable, but needs a real budget. Run stage 1 to 20k or more iterations (it had not yet recovered from the curriculum at 3k), and stage 2 until its episode length plateaus. Expect the late-clip falls to be partly this clip's fault too: it ends with a 24.7 degree pelvis tilt and the waist pitch at its limit.

## Step 8. Export

```bash
python deploy/export_policy.py --ckpt <stage-2 run>/model_4500.pt \
    --out rehearsal/policy/g1/asap_moving_arms/policy.pt --config deploy/policy_template/config.yaml --verify
```
0.7 s. Output: `weights: 8 tensors compared, all equal`; `outputs: 256 random inputs, max |policy.pt - checkpoint| = 5.7e-06 on outputs up to 16.3 (relative 3.5e-07)`; wrist outputs 0; `VERIFY PASSED`. Then the `config.yaml` top key was renamed to `g1/asap_moving_arms:` and `motion_length` set to 6.27.

Isaac Gym's ONNX export (`eval_agent.py`) was not run; it is a record-only artefact and the evaluation above used the same policy path.

## Hand-off for the MuJoCo gate

In the rehearsal hand-off folder, which is not published (22 MB, `SHA256SUMS` inside):

```
policy/g1/asap_moving_arms/{policy.pt, config.yaml, README.md}     rl_sar layout, motion_length 6.27
reference/0-motions_raw_tairantestbed_smpl_Eric_moving_arms_amass_grounded.pkl
checkpoints/stage1/{model_3000.pt, config.yaml}
checkpoints/stage2/{model_4500.pt, config.yaml}
eval/   curves, stills, contact sheets, TensorBoard summaries
```

| File | sha256 |
|---|---|
| `policy/g1/asap_moving_arms/policy.pt` | `cac0c4481b543b8ca7e6f50c1dca9741ee1854394fdba60f6a9e827cd84f3685` |
| `policy/g1/asap_moving_arms/config.yaml` | `baf25a58cd35fbf94247acb28e27ba0b70bbfc8a762d0cc5a3f92f0f60462079` |
| `checkpoints/stage1/model_3000.pt` | `bf9ab341fca5f8ecf0d74bc21e810f8ff63383d7599f396ec5fd34d6e69396b3` |
| `checkpoints/stage2/model_4500.pt` | `9f3c8496c3f0f532649fc25b0feae6ab656224b1ba35fc669c0a873cbe6885cb` |
| grounded pkl | `105180624c6d63ca03213101074dd1a31df3e4152296fee101c5a230d74a3d4f` |

The full run directories (every checkpoint, TensorBoard events, Hydra logs; 457 MB) stayed in a scratch clone of the fork and are not published. They are disposable.

## Reproducibility notes

- `import humanoidverse` resolved into the public clone before every Isaac Gym run (the stage scripts check it; outputs recorded in the logs).
- The public fork at `dab-v6-recipe` had every code and config file the two stages need: the reward `reward_motion_tracking_dm_2real_v5`, `off_trajectory_entry_fraction`, `max_learning_rate`, the NaN guard and the fixed `domain_rand_base.yaml`.
- Not on the fork, and not needed for training: `finetune_dab.sh` (v3; copied into `train/dab_as_run/` from the author's working copy), the frozen `domain_rand_v6_grounded.yaml` (identical values to `domain_rand_base.yaml`), and the author's uncommitted `eval_agent.py`, `sim2sim/` and converter scripts.
- `capture/convert_tram_to_asap.py` re-converted the TRAM results of Eric_moving_arms and wall_background_dab to npz files that are byte-identical (same sha256) to the originals. It must run in the `tram` env (numpy 2).
