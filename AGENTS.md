# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- This repo is glue: scripts, configs and docs. The code that trains and deploys lives in three repos pinned in `forks.lock` and cloned by `setup/bootstrap.sh`; change those by bumping the pin, never by copying their code here. The runbook is `docs/PIPELINE.md`.
- Isaac Gym runs: always headless with `DISPLAY` unset, `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`, the env's `bin` on `PATH` (ninja), and `PYTHONPATH=$ASAP_ROOT` so the intended ASAP checkout is imported; `train/stage*_*.sh` enforce all four. Checks: `docs/TROUBLESHOOTING.md`.
- Two Python envs: `hvgym` (Python 3.8, numpy 1.x: retarget, train, export, `capture/check_reference.py`, `capture/ground_reference.py`) and `tram` (numpy 2: TRAM and `capture/convert_tram_to_asap.py`, because TRAM saves results with numpy 2).
- Never commit videos, faces, other people's motions, SMPL model files or checkpoints over 10 MB (`setup/ASSETS.md`). Files derived from rl_sar stay Apache-2.0 with a notice of changes; list them in `NOTICE`.
- Tests: `bash tests/run_tests.sh` (CPU only, under a minute; registered as the no-mistakes test command in `.no-mistakes.yaml`). It needs the `hvgym` Python and fetches only the G1 robot model of the pinned ASAP commit when neither `$ASAP_ROOT` nor `external/ASAP` exists. Optional data-backed checks run when `DAB_V6_CKPT`, or `TRAM_RESULTS` with `CONVERT_EXPECTED_NPZ_DIR` (plus `CONVERT_PYTHON`, the `tram` env's python), are set (see the test files). `deploy/export_policy.py` always re-exports to `--out`, so never point `--out` at a shipped file; call `verify()` directly to check one.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
