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

RESULTS_PLACEHOLDER
