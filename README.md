# video2deploymentG1

From a phone video of you doing a move to a Unitree G1 humanoid policy that does the same move, checked in a second simulator and ready to try on the robot.

This repository holds the scripts, configurations and notes that glue together three existing projects: TRAM (video to human motion), ASAP (robot motion-tracking training with domain randomization) and rl_sar (the controller that runs on the G1). It pins each of them to an exact commit, so the same steps give the same result.

**Motions so far**
- **The dab**: taken from a phone video all the way to the real robot. Files, hashes and the full training history: [`motions/dab/`](motions/dab/README.md).
- **Rehearsal on `Eric_moving_arms`**: the two-stage training recipe, run end to end from the public fork, up to an exported policy. Both stages trained stably; with the short test budget (1.5 GPU-hours) the policy still falls in 4 of 10 simulated runs, so it is a plumbing test, not a usable policy. See [`docs/REHEARSAL_moving_arms.md`](docs/REHEARSAL_moving_arms.md).
- **The "6-7" gesture**: next. Recording notes in [`capture/RECORDING.md`](capture/RECORDING.md).

## The pipeline

```
 PHONE VIDEO     30 fps, landscape, tripod, whole body with feet, one person, ~6 s
     |           capture/RECORDING.md
     v
 TRAM            estimate_camera.py -> estimate_humans.py -> visualize_tram.py     (env: tram)
     |           out: tram/results/<name>/hps/hps_track_0.npy            ~2.5 min per clip
     v
 CONVERT         capture/convert_tram_to_asap.py <name> --fps 30 --out-dir /tmp/<name>_npz
     |           out: <name>_amass.npz (SMPL poses, shape, height)        seconds
     v
 RETARGET        ASAP scripts/data_process/fit_smpl_motion.py              (env: hvgym, CPU)
     |           out: ..._<name>_amass.pkl (G1 joint angles)              ~20 s
     v
 CHECK + GROUND  capture/check_reference.py, capture/ground_reference.py
     |           out: ..._<name>_amass_grounded.pkl (feet on the floor)   seconds
     v
 TRAIN           train/stage1_pretrain.sh        from scratch, domain randomization
     |           train/stage2_deploy_finetune.sh deployment fine-tune     (Isaac Gym, one GPU)
     |           out: model_N.pt                                          hours
     v
 EXPORT          deploy/export_policy.py --verify
     |           out: policy/g1/asap_<name>/{policy.pt, config.yaml}      seconds
     v
 SIM2SIM         rl_sar MuJoCo: the same controller code as the robot, walk -> motion -> walk
     |           pass = feet planted, heading steady, no fall, over 10+ trials
     v
 DEPLOY-READY    policy folder + FSM state in rl_sar, preflight passes.  STOP HERE.
```

## What you need

- A Linux machine (tested on Ubuntu 22.04) with an NVIDIA RTX-class GPU (tested on one RTX 4090) and conda.
- Downloads that this repository cannot redistribute: NVIDIA Isaac Gym Preview 4, the SMPL body model, TRAM's model weights. Where to register and what goes where: [`setup/ASSETS.md`](setup/ASSETS.md).
- For the last step only: a Unitree G1 and someone experienced with it. This repository stops before the robot.

## Setup

```bash
git clone https://github.com/EricXieQ/video2deploymentG1.git && cd video2deploymentG1
bash setup/bootstrap.sh ~/code/g1        # clones ASAP fork, rl_sar fork and TRAM at the commits in forks.lock
export ASAP_ROOT=~/code/g1/ASAP TRAM_ROOT=~/code/g1/tram RL_SAR_ROOT=~/code/g1/rl_sar
conda env create -f setup/env_hvgym.yml  # training env; then the pip install -e lines at the top of the file
conda env create -f setup/env_tram.yml   # video-to-motion env
```

Pinned commits ([`forks.lock`](forks.lock)):

| Repository | Ref | Commit |
|---|---|---|
| ASAP fork https://github.com/EricXieQ/ASAP | tag `dab-v6-recipe` (branch `dab-recipe-v6`) | `af60acf` |
| rl_sar fork https://github.com/EricXieQ/rl_sar_g1_deployment | branch `jetson-deploy` | `314b2cb` |
| TRAM https://github.com/yufu-wang/tram | `main` | `4861c11` |

Check the install with the repository's own tests (CPU only, under a minute; uses the `hvgym` env):

```bash
bash tests/run_tests.sh
```

## Walkthrough

The full runbook, with every command, its duration and how to check it, is [`docs/PIPELINE.md`](docs/PIPELINE.md). In short:

1. **Record** the move (`capture/RECORDING.md`).
2. **TRAM**: three commands turn the video into SMPL motion.
3. **Convert**: `python capture/convert_tram_to_asap.py <name> --fps 30 --out-dir /tmp/<name>_npz` (a staging folder; step 4 copies from it).
4. **Retarget** to the G1 with ASAP's `fit_smpl_motion.py` (it overwrites every clip in its folder: isolate yours first).
5. **Check and ground**: `check_reference.py` flags floating feet, lean, fast or limit-bound joints; `ground_reference.py` puts the lower foot on the floor in every frame.
6. **Train stage 1**: `MOTION_PKL=... PROJECT=... ITERATIONS=20000 bash train/stage1_pretrain.sh`.
7. **Train stage 2**: `CHECKPOINT=<stage-1 model> ... bash train/stage2_deploy_finetune.sh`.
8. **Export**: `python deploy/export_policy.py --ckpt <model> --out <rl_sar>/policy/g1/asap_<name>/policy.pt --verify`.
9. **Integrate** into rl_sar's state machine (`deploy/FSM_INTEGRATION.md`).
10. **MuJoCo gate**: 10 or more walk, motion, walk trials, scored.
11. **Deploy-ready checklist**, then stop.

## Expected cost

On one RTX 4090: minutes for each capture step; about 5 to 12 h for stage 1 (20k to 40k iterations at roughly 3,500 per hour) and 1 to 5 h for stage 2. Each checkpoint is 9.8 MB, one every 100 iterations.

## How to judge a policy

"It did not fall" is not enough: that once hid a policy that stepped and turned 90 degrees. In MuJoCo, over 10 or more fresh trials of walk, motion, walk: both feet stay planted (no lift of 1 cm or more for 80 ms after the first 0.5 s), the heading at the end is within a few degrees of the start, torso roll stays small, nothing aborts, and the return to walking is planted too.

## What differs from the ASAP paper

Broader domain randomization than the paper in most terms, with control delay widened to 0 to 60 ms to cover the 50 ms measured on the robot; training in two stages; the paper's physics-cleaning step (MaskedMimic) skipped; ASAP's delta-action stage not used. Details: [`docs/PAPER_VS_THIS.md`](docs/PAPER_VS_THIS.md). Known pitfalls: [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

## Safety and disclaimer

Humanoid robots are heavy, fast and can hurt people and themselves. Everything here is research code provided as is, without warranty. **This repository stops at "deploy-ready"**: a policy that passed the MuJoCo checks and a controller that builds. Running it on a real robot is your responsibility and needs a hoist or gantry, an emergency stop within reach, a clear area, and someone who knows the robot. As the ASAP authors also warn, do not run these policies on hardware without that expertise.

## Credits and licences

This repository: MIT ([`LICENSE`](LICENSE)), except the files derived from rl_sar, which stay under Apache-2.0 with a notice of changes ([`NOTICE`](NOTICE)).

Built on, and please cite:
- **ASAP**: T. He et al., "ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills", RSS 2025, arXiv:2502.01143. Code (MIT): https://github.com/LeCAR-Lab/ASAP, built on HumanoidVerse.
- **TRAM**: Y. Wang et al., "TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos", ECCV 2024. Code (MIT): https://github.com/yufu-wang/tram.
- **PHC** (retargeting approach and SMPLSim): Z. Luo et al., "Perpetual Humanoid Control for Real-time Simulated Avatars", ICCV 2023.
- **rl_sar** (Apache-2.0): https://github.com/fan-ziqi/rl_sar.
- **SMPL**: M. Loper et al., "SMPL: A Skinned Multi-Person Linear Model", SIGGRAPH Asia 2015 (research licence; not redistributed).
- **Isaac Gym**: NVIDIA (not redistributed).

No videos or faces are published in this repository.
