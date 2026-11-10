# Pitfalls and how to avoid them

Each of these cost real time on the dab.

## Setup

1. **The wrong ASAP gets imported.** With more than one ASAP checkout (a worktree, a fork clone), `pip install -e` points `import humanoidverse` at whichever was installed last. Set `PYTHONPATH=$ASAP_ROOT` and check with `python -c "import humanoidverse; print(humanoidverse.__file__)"`. The stage scripts do both and refuse to start on a mismatch.
2. **`libpython3.8.so.1.0: cannot open shared object file`.** Isaac Gym links against the conda env's Python library: `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`.
3. **`RuntimeError: Ninja is required to load C++ extensions`.** gymtorch compiles on first import and needs `ninja` from the env on `PATH`. `conda activate hvgym` provides it; when calling the env's python by full path, add `$CONDA_PREFIX/bin` to `PATH` (the stage scripts do).
4. **`ModuleNotFoundError: No module named 'numpy._core'` in the converter.** TRAM saves its results with numpy 2; the `hvgym` env has numpy 1.23. Run `convert_tram_to_asap.py` in the `tram` env (it also needs scipy there).
5. **Never open an Isaac Gym window on a shared or remote machine.** Pass `headless=True` (the default in the stage scripts) and unset `DISPLAY`.

## Capture and reference motion

6. **Wrong fps label** changes the motion's speed. Always pass the true video rate to the converter.
7. **Smoothing eats fast motion.** The converter's default 5-frame box filter keeps 83 % of a 2 Hz oscillation and 65 % of a 3 Hz one. Use `--smooth 3` or `0` for fast gestures.
8. **Floating feet.** TRAM's per-frame height is noisy and the retargeter grounds one frame only; ASAP's own height fix is disabled (`motion_lib_base.py`, `fix_height = False`). The dab's lower foot floated 1 to 6 cm above the floor (5 cm range), Eric_moving_arms 1 to 11 cm. Fix: `capture/ground_reference.py`.
9. **Forward lean.** Every captain clip retargets with an 18 to 27 degree pelvis tilt (most of the paper's clips: 4 to 15). The SMPL pelvis from TRAM is close to vertical and the camera was level, so the tilt comes from retargeting; the root cause is not established (possibly the spine and neck posture TRAM estimates). On Eric_moving_arms the waist pitch joint also sits at its 0.52 rad limit for about 1 s. `check_reference.py` warns; grounding does not fix it.
10. **The retarget batch overwrites everything.** `fit_smpl_motion.py` re-retargets every npz in `raw_tairantestbed_smpl/` and overwrites their pkls. Isolate the new npz first.
11. **`--lock_height` with bent knees** lifts the feet. Do not use it for bounces or squats; ground per frame instead.

## Training

12. **Runs die.** An X crash killed one dab fine-tune; another died with no log. Use tmux and `tee` (the stage scripts tee to a log file).
13. **NaN blow-ups.** The dab's v5 died on a NaN after about 30k iterations. Keep the fork's NaN guard, the learning-rate cap (`+algo.config.max_learning_rate=3.e-4`) and the gradient clip (`algo.config.max_grad_norm=0.5`).
14. **No stopping rule.** The dab's base run went to 93,600 iterations because nobody decided when to stop. Look at TensorBoard `Train/mean_episode_length` and an evaluation at 10k, 20k and 40k.
15. **Penalizing the mechanism of recovery** destroys it: the dab's v7 added foot-motion penalties and could no longer recover from pushes.
16. **Fine-tune iteration numbers continue from the checkpoint.** Resuming `model_3000.pt` for 1500 iterations writes `model_3000.pt` (again, in the new run folder) to `model_4500.pt`.

## Export and deployment

17. **Observation order.** ASAP concatenates observation terms in alphabetical key order (`legged_robot_base.py`). Any new runner must match; a wrong order makes the robot spin.
18. **Export defaults.** The original export script silently re-exported an old checkpoint over a newer `policy.pt` when run without arguments. `deploy/export_policy.py` requires `--ckpt` and `--out`, and `--verify` checks the result.
19. **History off by one.** The shipped dab wrapper feeds rl_sar frames t..t-3 into ASAP's history, where training used t-1..t-4 ("convention A"). A corrected wrapper ("convention B", rl_sar branch `fix/deploy-history-contract-2026-10`) matches training but failed the walk-dab-walk MuJoCo test. `export_policy.py` keeps A so new policies match the deployed dab; the open question is tracked in rl_sar.
20. **Motion length is compiled in.** rl_sar's dab state sets `rl.motion_length = 5.93f` in `fsm_g1.hpp`. A policy for a clip of another length gets a wrong phase clock unless the FSM changes: `deploy/FSM_INTEGRATION.md`.
21. **The hand-off from walking.** Enter the motion policy directly and keep it alive; pre-positioning stages failed. Train with off-trajectory entries (stage 2): without them the dab's first tick demanded 1.96 rad. Keep the hold-pose seed, the 160 ms entry ramp (`interp_entry_steps: 32`), the 0.7 action smoothing and the walking policy's 20-tick return ramp.
22. **MuJoCo viewer:** press `R` after `0` in a free-standing scene, or the robot spawns collapsed. Use `scene_29dof`, not `hoist2` (its ropes make the left knee buzz). One `1` then the motion key reproduces only the first rep of a robot process; relaunch for clean trials.
23. **Build:** plain `./build.sh` in rl_sar runs colcon and leaves a stale MuJoCo binary; use `./build.sh -mj` (MuJoCo) or `-m` (robot).
24. **Gain percentage:** older rl_sar builds start at 60 % action scale and kp; check with `+` and `]`. `jetson-deploy` starts at 100 %.
25. **Command delay on the robot is about 50 ms**, and the control loop runs about 2.5 % slow (48.8 Hz instead of 50). Training now randomizes delay up to 60 ms (docs/PAPER_VS_THIS.md).
