# The pipeline, step by step

From a phone video to a policy that is ready to try on a Unitree G1. Each step lists what to run, how long it takes on one RTX 4090, and how to check it worked. Steps marked **new** are ones the dab skipped and should not have.

Terms used below:
- **SMPL**: a standard 3D human body model (24 joint rotations, 10 body-shape numbers, a root position).
- **Reference motion**: the robot-joint version of the human motion, which the policy learns to copy.
- **Phase**: a clock from 0 to 1 over the clip. It is the policy's only "what to do now" signal; no motion file is loaded on the robot.
- **Iteration**: one PPO update after 24 simulated steps of 4096 robots.
- **Domain randomization (DR)**: random friction, motor gains, masses, pushes and delays during training, so the policy survives physics it has not seen.

Environment variables used throughout: `ASAP_ROOT`, `TRAM_ROOT`, `RL_SAR_ROOT` (from `setup/bootstrap.sh`), and `V2D` for this repository.

## Step 0. Environment (once)

```bash
bash setup/bootstrap.sh ~/code/g1          # clones the three pinned repos (forks.lock)
export ASAP_ROOT=~/code/g1/ASAP TRAM_ROOT=~/code/g1/tram RL_SAR_ROOT=~/code/g1/rl_sar V2D=$PWD
conda env create -f setup/env_hvgym.yml    # then the three pip install -e lines at the top of that file
conda env create -f setup/env_tram.yml     # or follow TRAM's own install
```
Then download the files in `setup/ASSETS.md`. Build rl_sar for MuJoCo with `./build.sh -mj` (see its README).

- **Check:** `python -c "import humanoidverse; print(humanoidverse.__file__)"` prints a path inside `$ASAP_ROOT`.
- **Pitfall:** with several ASAP checkouts on one machine, the editable install imports whichever was installed last. The training scripts set `PYTHONPATH=$ASAP_ROOT` and refuse to start if the import resolves elsewhere.
- Every Isaac Gym process needs `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` (and `ninja` on `PATH` the first time gymtorch compiles). The scripts set both.

## Step 1. Record the video

See `capture/RECORDING.md`: 30 fps, tripod, level, whole body with feet, one person, start and end in a walking-like stance, 4 to 7 s.

- **Time:** minutes.
- **Check:** `ffprobe` frame rate and frame count.

## Step 2. TRAM: video to SMPL

```bash
conda activate tram && cd $TRAM_ROOT
python scripts/estimate_camera.py --video /path/to/my_motion.mov   # frames, detection, masked DROID-SLAM
python scripts/estimate_humans.py --video /path/to/my_motion.mov   # VIMO -> results/my_motion/hps/hps_track_0.npy
python scripts/visualize_tram.py  --video /path/to/my_motion.mov   # results/my_motion/tram_output.mp4
```
- **Time:** about 2.5 min for a 6 s clip.
- **Check:** watch `tram_output.mp4`: feet on the floor, no swapped limbs, one person.
- Whether the dab used `--static_camera` is unknown; TRAM's camera estimate for it was static and level anyway.

## Step 3. Convert to an ASAP SMPL file

```bash
python $V2D/capture/convert_tram_to_asap.py my_motion --fps 30 --smooth 5 \
    --tram-results $TRAM_ROOT/results --out-dir /tmp/my_motion_npz
```
It reads TRAM's camera-frame track, turns it 90 degrees to Z-up, zeroes X and Y travel, keeps the height, box-smooths poses and translation, averages the body shape, and writes `my_motion_amass.npz`.

- **Pitfalls:**
  - The `--fps` label sets the speed. Use the true video rate.
  - The default 5-frame smoothing removes 17 % of a 2 Hz oscillation and 35 % of a 3 Hz one. Use `--smooth 3` or `0` for fast motions.
  - Do not use `--lock_height` for anything that bends the knees.
  - It needs numpy 2 (TRAM's results are saved with it) plus torch and scipy: run it in the `tram` env.
- **Time:** seconds.
- **Idea, not done yet:** converting from TRAM's gravity-aligned world frame (`camera.npy` `world_cam_R/T`) instead of the fixed 90 degree turn would remove the level-camera assumption.

## Step 4. Retarget to the G1

```bash
conda activate hvgym && cd $ASAP_ROOT            # must be the ASAP root: the output name depends on the path
mkdir -p humanoidverse/data/motions/raw_tairantestbed_smpl
cp /tmp/my_motion_npz/my_motion_amass.npz humanoidverse/data/motions/raw_tairantestbed_smpl/
python scripts/data_process/fit_smpl_shape.py  +robot=g1/g1_29dof_anneal_23dof   # once per robot
python scripts/data_process/fit_smpl_motion.py +robot=g1/g1_29dof_anneal_23dof
python scripts/vis/vis_q_mj.py +robot=g1/g1_29dof_anneal_23dof \
    +visualize_motion_file=humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_my_motion_amass.pkl
```
- **Trap:** `fit_smpl_motion.py` re-retargets **every** npz in `raw_tairantestbed_smpl/` and overwrites the existing pkls. The pinned fork ships 51 of ASAP's own clips there. Back up `singles/`, or move the other npz files out while you retarget.
- It optimizes joint angles for 1000 steps to match 14 joint positions, smoothing after each step, and grounds **one** frame only (the lowest one).
- **Time:** about 10 to 25 s per clip on the CPU.
- **Check:** `vis_q_mj.py` playback.

## Step 5. Check and ground the reference (new)

```bash
PKL=$ASAP_ROOT/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_my_motion_amass.pkl
python $V2D/capture/check_reference.py  --asap-root $ASAP_ROOT $PKL
python $V2D/capture/ground_reference.py --asap-root $ASAP_ROOT $PKL     # writes ..._grounded.pkl beside it
python $V2D/capture/check_reference.py  --asap-root $ASAP_ROOT ${PKL%.pkl}_grounded.pkl
```
`check_reference.py` prints PASS or WARN for: fps; length; the per-frame lowest-foot height (range within 1 cm, within 1 cm of the 0.035 m floor constant); pelvis tilt (mean under 15 degrees); peak joint speed against limits; time spent at joint limits; start and end pose against the walking stance.

`ground_reference.py` shifts the whole body vertically on every frame so the lower ankle link sits exactly at 0.035 m (the height of that link when the G1 stands flat-footed). Rotations, joint angles and timing are untouched.

- **Pass:** lowest foot constant within about 1 cm; pelvis tilt under about 15 degrees when standing quietly; no joint parked at a limit.
- Grounding fixes floating feet. It does **not** fix a forward lean: every captain clip so far retargets with an 18 to 27 degree pelvis tilt, cause not established. Train anyway if the motion looks right; note the WARN.
- **Time:** seconds.

## Step 6. Training, stage 1 from scratch

```bash
export ASAP_ROOT CONDA_ENV_PREFIX=$CONDA_PREFIX
MOTION_PKL=${PKL%.pkl}_grounded.pkl PROJECT=MyMotion EXPERIMENT=MyMotion_stage1 ITERATIONS=20000 \
  bash $V2D/train/stage1_pretrain.sh
```
The dab's verified v1 command with domain randomization on, plus the widened control delay (0 to 60 ms). Logs and checkpoints go to `$ASAP_ROOT/logs/MyMotion/<time>-MyMotion_stage1-.../model_N.pt` (one per 100 iterations, 9.8 MB each); the console log goes next to them.

- Run it in tmux. A dropped X session killed one dab run; another died without a log.
- **Time:** about 3,300 to 4,000 iterations per hour. The dab ran 93,600 iterations (42 h) with no stopping rule. Budget 20k to 40k (5 to 12 h) and look at 10k, 20k and 40k.
- **Check:** TensorBoard (`tensorboard --logdir $ASAP_ROOT/logs/MyMotion`), `Train/mean_episode_length` rising towards the clip's step count, and `eval_agent.py` (step 8) looks right.

## Step 7. Training, stage 2: deployment fine-tune

```bash
MOTION_PKL=${PKL%.pkl}_grounded.pkl PROJECT=MyMotion EXPERIMENT=MyMotion_stage2 ITERATIONS=5000 \
CHECKPOINT=$ASAP_ROOT/logs/MyMotion/<stage1-run>/model_20000.pt \
  bash $V2D/train/stage2_deploy_finetune.sh
```
One run with the dab v6's whole flag set (reward v5, 20 % off-trajectory entries from the walking stance, clean spawns, learning rates 1e-4 / 2e-4 with a 3e-4 cap, gradient clip 0.5). It replaces the dab's four cumulative fine-tunes. First tested on the Eric_moving_arms rehearsal; see `docs/REHEARSAL_moving_arms.md` for how that went.

- **Time:** the dab's v6 reached its deployed checkpoint after 20,000 iterations (5 h). Fine-tunes were seen to saturate after 1k to 5k iterations.
- **Checks:** episode length must not collapse and no NaN; entries from the walking stance look clean in `eval_agent.py`; a push test (new: the dab's v7 failed exactly this and nobody checked until later).
- **Do not** add foot-motion penalties (the dab's v7 did and lost push recovery).

## Step 8. Export

```bash
cd $ASAP_ROOT
python humanoidverse/eval_agent.py +checkpoint=logs/MyMotion/<stage2-run>/model_N.pt   # Isaac Gym look; also writes exported/model_N.onnx
python $V2D/deploy/export_policy.py --ckpt logs/MyMotion/<stage2-run>/model_N.pt \
    --out $RL_SAR_ROOT/policy/g1/asap_my_motion/policy.pt \
    --config $V2D/deploy/policy_template/config.yaml --verify
```
`eval_agent.py` opens a viewer by default (`headless=False`, 1 robot); pass `headless=True` on a machine without a display. `export_policy.py` wraps the checkpoint's actor as a TorchScript `policy.pt` that takes rl_sar's 376 inputs. `--verify` checks every weight tensor and compares the exported policy with an independent re-implementation on random inputs.

Then rename the top-level key of the copied `config.yaml` to `g1/asap_my_motion:`.

- **History convention:** the wrapper keeps the dab's shipped layout ("A"); a corrected layout ("B") exists on an rl_sar branch but failed its MuJoCo test. See `docs/TROUBLESHOOTING.md`.

## Step 9. rl_sar integration

- New folder `$RL_SAR_ROOT/policy/g1/asap_my_motion/` with `policy.pt`, `config.yaml` and a README (checkpoint path, sha256, scores).
- Set `motion_length` in that `config.yaml` to the clip's `(frames - 1) / 30` seconds (the "length" line of `check_reference.py`) and select the folder with `RL_DAB_POLICY=asap_my_motion`. This needs an rl_sar build with the config-driven length; on the pinned 314b2cb it is still compiled in as 5.93 s. See `deploy/FSM_INTEGRATION.md`.
- Rebuild with `cmake --build cmake_build --target rl_sim_mujoco`.

## Step 10. MuJoCo gate (sim2sim)

```bash
cd $RL_SAR_ROOT && ./cmake_build/bin/rl_sim_mujoco g1 scene_29dof
# keys: 0, wait about 3 s, R, 1 (walking), wait 4.5 to 6 s, then the motion's key
```
Repeat 10 or more times, quitting and relaunching for clean trials. Score with rl_sar's `scripts/score_dab_session.py`. Its default windows are the dab's; with the config-driven-length rl_sar commit, pass `--motion-length <s> --win-raise <a> <b> --win-lean <a> <b>` for another motion.

- **Pass ("the bar"):** feet planted in all or nearly all trials (no lift of 1 cm or more for 80 ms after the first 0.5 s), endpoint heading within a few degrees, no abort, and the return to walking also planted. "Never falls" is not enough: it once hid a policy that stepped and turned 90 degrees.
- Use `scene_29dof` for scoring; the `hoist2` ropes cause a left-knee buzz.

## Step 11. Deploy-ready checklist, then stop

- Policy folder committed on a branch of your rl_sar fork.
- FSM build verified (`strings cmake_build/bin/rl_sim_mujoco | grep INITRL`), preload lines seen.
- Cheat-sheet entry for the new key and folder.
- The first-tick jolt table checked in MuJoCo.
- `scripts/preflight.sh` passes; scoring windows set.

**Stop here.** Running on the real robot is a separate, supervised step with a hoist, an emergency stop and someone who knows the robot. See rl_sar's `JETSON_SETUP.md` and `reports/HW_SESSION_2026-09.md` on the pinned commit.

## Everywhere the dab was hard-coded

| Where | Dab value | For a new motion |
|---|---|---|
| converter paths | absolute paths into one home folder | now `--tram-results` and `--out-dir` |
| `fit_smpl_motion.py` | every npz in one folder | isolate the new npz |
| training motion file, project, experiment | `..._wall_background_dab_amass.pkl`, `DabTracking` | the stage scripts take variables |
| `motion_tracking.py` `LOCO_STANCE_ASAP_23DOF`, entry height 0.78 m | the walking policy's stance | same, if you enter from the same walking policy |
| termination: tilt 0.95, minimum height 0.3 m | | recheck for squats or deep leans |
| export defaults | v6 checkpoint into `asap_dab` | `export_policy.py` requires `--ckpt` and `--out` |
| `config.yaml` top key | `g1/asap_dab:` | `g1/asap_<name>:` |
| `fsm_g1.hpp` key 5, `asap_dab`, 5.93 s | | `deploy/FSM_INTEGRATION.md` |
| `fsm_g1.hpp` 30 degree abort | | keep; check the motion's tilt stays well below it |
| `score_dab_session.py` windows | dab windows | new windows |

## Cost

Minutes per capture step. Stage 1 about 5 to 12 h, stage 2 about 1 to 5 h, on one RTX 4090 with 4096 robots. Disk: 9.8 MB per checkpoint, one every 100 iterations (about 2 GB per 20k iterations).
