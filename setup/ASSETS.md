# Files you must download yourself

This repository redistributes none of the files below. Each one has its own licence, and most need a free registration. Download them once, put them where shown, and the pipeline finds them.

`$ASAP_ROOT`, `$TRAM_ROOT` and `$RL_SAR_ROOT` are the clones made by `setup/bootstrap.sh`.

| What | Needed for | Where to get it | Where it goes |
|---|---|---|---|
| **Isaac Gym Preview 4** | training, evaluation (steps 6 to 8) | NVIDIA, after accepting its licence: https://developer.nvidia.com/isaac-gym | anywhere; then `pip install -e <isaacgym>/python` in the `hvgym` env |
| **SMPL body model** v1.1.0, pkl format | retargeting (step 4) | register at https://smpl.is.tue.mpg.de, download `SMPL_python_v.1.1.0.zip` | unzip into `$ASAP_ROOT/humanoidverse/data/smpl/` so it holds `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl`, `SMPL_FEMALE.pkl` (ASAP README, "SMPL Shape preparation") |
| **TRAM models and checkpoints** (about 5.6 GB, includes SMPL again) | video to SMPL (step 2) | register at https://smplify.is.tue.mpg.de and https://smpl.is.tue.mpg.de, then `cd $TRAM_ROOT && bash scripts/download_models.sh` | `$TRAM_ROOT/data/` (the script does it) |
| **G1 SMPL shape fit** `shape_optimized_v1.pkl` | retargeting (step 4) | make it: `python scripts/data_process/fit_smpl_shape.py +robot=g1/g1_29dof_anneal_23dof` from `$ASAP_ROOT`, once per robot | `$ASAP_ROOT/humanoidverse/data/shape/g1_29dof_anneal_23dof/` |

Licences in short:
- **SMPL** is a research licence that forbids redistributing the model files. ASAP's `.gitignore` excludes them, and so does this repository.
- **Isaac Gym** is under an NVIDIA licence and is not redistributable.
- **TRAM's code** is MIT, but the models it downloads (SAM, Detectron2, DEVA, DROID-SLAM, ZoeDepth, the VIMO checkpoint) have their own licences.
- **`smpl_sim`** (PHC's helper, used by retargeting) declares no licence in its package metadata. This repository only installs it (`setup/env_hvgym.yml`) and copies nothing from it.

What this repository does ship: the dab's SMPL parameter file (`motions/dab/wall_background_dab_amass.npz`, the captain's own motion, no body-model data), its retargeted robot motion, and its exported policy. See `motions/dab/README.md`.
