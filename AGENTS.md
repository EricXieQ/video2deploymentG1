# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- This repo is glue: scripts, configs and docs. The code that trains and deploys lives in three repos pinned in `forks.lock` and cloned by `setup/bootstrap.sh`; change those by bumping the pin, never by copying their code here. The runbook is `docs/PIPELINE.md`.
- Isaac Gym runs: always headless with `DISPLAY` unset, `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`, the env's `bin` on `PATH` (ninja), and `PYTHONPATH=$ASAP_ROOT` so the intended ASAP checkout is imported; `train/stage*_*.sh` enforce all four. Checks: `docs/TROUBLESHOOTING.md`.
- Two Python envs: `hvgym` (Python 3.8, numpy 1.x: retarget, train, export, `capture/check_reference.py`, `capture/ground_reference.py`) and `tram` (numpy 2: TRAM and `capture/convert_tram_to_asap.py`, because TRAM saves results with numpy 2).
- Never commit videos, faces, other people's motions, SMPL model files or checkpoints over 10 MB (`setup/ASSETS.md`). Files derived from rl_sar stay Apache-2.0 with a notice of changes; list them in `NOTICE`.
- No test suite yet. Regression checks used so far: re-convert a TRAM result and compare the npz (`capture/convert_tram_to_asap.py`), `deploy/export_policy.py --verify` on `motions/dab/policy/policy.pt` against the dab v6 checkpoint, and `check_reference.py` numbers for the dab pkl in `motions/dab/README.md`.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
