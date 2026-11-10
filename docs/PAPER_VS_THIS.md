# What differs from the ASAP paper

ASAP: "Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills", He et al., 2025, arXiv:2502.01143. This project uses the ASAP code's phase-based motion tracking (the paper's stage 1, "pre-training"). It does not use ASAP's delta-action model (the paper's stage 2, "post-training"), and it differs in the ways below.

## Domain randomization

The paper's Table VI against what the dab ran (v1 to v6, `config/domain_rand/domain_rand_base.yaml` in the fork) and what this repository's stage scripts run.

| Term | Paper, Table VI | Dab (v1 to v6) | This repo (stage 1 and 2) |
|---|---|---|---|
| Friction | U(0.2, 1.1) | U(0.5, 1.25) | same as dab |
| Motor gains | P gain U(0.925, 1.05) x default | kp and kd each U(0.75, 1.25) x default | same as dab |
| **Control delay** | U(20, 40) ms | 0, 1 or 2 policy steps = 0, 20 or 40 ms | **0, 1, 2 or 3 steps = 0 to 60 ms** |
| Pushes | every 10 s, 0.5 m/s | every 5 to 10 s, up to 1.0 m/s | same as dab |
| Base centre-of-mass shift | not in Table VI | ±0.1 m per axis | same as dab |
| Link mass | not in Table VI | x U(0.8, 1.2) | same as dab |
| Random torque noise | not in Table VI | 0.1 x torque limit, scaled U(0.5, 1.5) | same as dab |

So the dab randomized more broadly than the paper in every term except the lowest friction. ASAP's own README trains with randomization off (`+domain_rand=NO_domain_rand`); turning it on was the captain's choice.

### Deliberate deviation: wider control delay

On the real G1 a command first acts about 50 ms after it is sent (measured from state logs). The dab trained with at most 40 ms. This repository's stage scripts set `domain_rand.ctrl_delay_step_range=[0,3]`. One policy step is 20 ms (200 Hz simulation, decimation 4), so the range is 0 to 60 ms. Each robot draws one delay per episode (`legged_robot_base.py`, `action_queue`). Set `CTRL_DELAY_MAX=2` to reproduce the dab exactly.

The comment in `domain_rand_base.yaml` ("max real delay is 90ms") is wrong: with `[0, 2]` the maximum was 40 ms.

## Training schedule

| | Paper / ASAP README | Dab | This repo |
|---|---|---|---|
| Stages | one motion-tracking run | v1 from scratch (93,600 iterations, 42 h), then four cumulative fine-tunes v3 to v6 (about 28 h more) | stage 1 from scratch, stage 2 = v6's final flag set applied once |
| Termination curriculum (1.5 m tightening to 0.3 m) | on | on in v1, off in v3 to v6 | on in stage 1, off in stage 2 |
| Start state | reference state initialization (random phase) | same; since v3, 20 % of episodes start at phase 0 from the walking stance | same as dab v6 |
| Rewards | Table I weights | stock until v3; v4 adds an ankle command-over-limit penalty, v5 tightens tracking tolerances | stage 1 stock, stage 2 reward v5 |
| PPO safeguards | none | v6: NaN guard, learning-rate cap 3e-4, gradient clip 0.5 | stage 2 as v6 (the NaN guard is in the fork's code for both stages) |

Unchanged from the paper: asymmetric actor-critic, the phase input, the 5-frame proprioception history (4 past frames plus the current one), the network sizes.

## Steps skipped

- **Physics cleaning of the captured motion.** The paper runs every TRAM motion through MaskedMimic in simulation and keeps only motions a physics character can perform (section II-A-b). The dab skipped it, and so does this repository, by decision. `capture/check_reference.py` and `capture/ground_reference.py` catch the two defects that hurt the dab (floating feet, lean); they are not a physics filter.
- **Delta-action model (ASAP stage 2).** Not used. Sim-to-real gaps are handled with domain randomization and the deploy-side settings in rl_sar.
- **Reference sanity checks, a standard push test, multi-trial sim2sim scoring before hardware, and verifying the export against the checkpoint** were missing for the dab. The first and last are now scripts here (`check_reference.py`, `export_policy.py --verify`); the scoring exists in rl_sar (`scripts/score_dab_session.py`); a push test is still manual.
