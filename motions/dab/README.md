# The dab

The first motion taken through this pipeline, from a phone video to a policy that ran on a real Unitree G1 (the entry stomp was gone on hardware on 2026-07-18; 69 and 404 repetitions were captured on 2026-07-30 and 2026-08-02).

Clip: `wall_background_dab`, 1280x720 at 30 fps, 179 frames (5.97 s of video; 5.93 s of motion as ASAP counts it). No video is published.

## Files

| File | What it is | sha256 |
|---|---|---|
| `wall_background_dab_amass.npz` | SMPL motion from TRAM, made by `capture/convert_tram_to_asap.py` with defaults (`--fps 30 --smooth 5`) | `97a9763e59511c83e30139ca4482575366569b82bd01a4bedb79113788ace7cb` |
| `0-motions_raw_tairantestbed_smpl_wall_background_dab_amass.pkl` | the retargeted G1 reference motion every training stage used | `7b637ddc4443be9ccaf231b007a75443de3e8b41da8b8770aa76f641d657648b` |
| `train_config_v6.yaml` | the full Hydra config of the deployed run (v6), as saved by ASAP | `f8f768529fedece5221bc55100d68e8650c70d1ebf8a9568a20311b1c709ccd1` |
| `policy/policy.pt` | v6 `model_20000` exported as rl_sar's TorchScript wrapper (from rl_sar `policy/g1/asap_dab/` at 314b2cb) | `4adb4a81a529923a7b01984a0473d9fc4f706d30542b3e9a2f0e403542962437` |
| `policy/config.yaml` | the rl_sar policy config that runs it (same source) | `fceeadc1bd28141abf5f59de5429d48ec567922fb882c935060f21586e1319a0` |

`policy/` comes from rl_sar and is under the Apache License 2.0 (`deploy/LICENSE-APACHE-2.0`); it is copied unchanged.

Checks done when building this repository:
- `policy.pt` passes `deploy/export_policy.py`'s verifier against the v6 checkpoint (8 of 8 weight tensors equal; outputs match an independent re-implementation to 2e-7 relative). Re-exporting the checkpoint with `export_policy.py` gives a policy with identical outputs.
- Converting TRAM's result for this clip again with `capture/convert_tram_to_asap.py` reproduces the npz byte for byte.
- `capture/check_reference.py` on the pkl: lowest foot 4.5 to 9.7 cm above the floor (5.2 cm range), pelvis tilt 22 degrees on average. The dab was trained on this floating, leaning reference.

The checkpoint itself (`model_20000.pt`, 9.8 MB with optimizer state) is not in the repository.

## Training lineage

All runs: Isaac Gym, 4096 robots, one RTX 4090, `config/domain_rand/domain_rand_base.yaml` (control delay 0 to 40 ms). Commands: `train/dab_as_run/` (v3 to v6 verbatim) and the v1 command in `docs/PIPELINE.md` (stage 1 here is that command plus a wider delay).

| Stage | Run directory (ASAP `logs/DabTracking/`) | Starts from | What changed | Checkpoint used / run length | GPU hours to that checkpoint |
|---|---|---|---|---|---|
| no-DR trial | `20260323_215341-..._wall_background` | scratch | ASAP README recipe, randomization off | 2,800 | 0.8 |
| **v1 base** | `20260328_172005-..._wall_domainrand` | scratch | README recipe plus `+domain_rand=domain_rand_base` | 93,600 / 93,601 | 42.0 |
| v3 | `20260504_015435-..._wall_offtraj_finetune_v3` | v1 93,600 | off-trajectory entry 20 %, init noise 2, tilt and low-height termination, learning rates 1e-4 / 2e-4, motion-far termination off | 25,600 / 25,611 | 7.1 |
| v4 | `20260716_131145-..._ankleyaw_v4` | v3 25,600 | reward `..._ankleyaw`: ankle command-over-limit penalty -10 | 41,600 / 41,665 | 11.0 |
| v5 | `20260717_013225-..._spineyaw_v5` | v4 41,600 | reward `..._v5`: joint and body-rotation tolerances 1.0 to 0.25 | 6,700 (run died, cause unknown) | 1.7 |
| v5 resume | `20260717_124339-..._spineyaw_v5_resume` | v5 6,700 | same | 13,000 / 29,925 (died later on a NaN) | 3.4 |
| **v6 (deployed)** | `20260718_015831-..._grounded_v6` | v5 resume 13,000 | init noise 2 to 0.25, learning-rate cap 3e-4, gradient clip 0.5 | **20,000** / 52,054 | 5.1 |
| v7 (shelved) | `20260718_150941-..._reflex_v7` | v6 20,000 | foot penalties; could not recover from pushes | 4,800 | |

About 70 GPU-hours up to the deployed checkpoint, not counting dead ends. ("v2" was a pair of short trials that left no run directory.)

In MuJoCo (rl_sar, scripted trials) v6 scored: feet planted 10 of 10, endpoint heading +1.3 degrees, peak torso roll -3.0 degrees.
