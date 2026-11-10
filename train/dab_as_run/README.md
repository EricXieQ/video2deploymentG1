# The dab's fine-tune scripts, exactly as run

Verbatim copies, kept as a record. Use `train/stage1_pretrain.sh` and `train/stage2_deploy_finetune.sh` for new motions. These scripts hold absolute paths of the machine they ran on and checkpoint paths that only exist there.

| File | Stage | Warm start | Source |
|---|---|---|---|
| `finetune_dab.sh` | v3: off-trajectory entry | v1 `model_93600` | captain's ASAP working copy (untracked there; **not** on the public fork's tag) |
| `finetune_dab_v4.sh` | v4: ankle command-over-limit penalty | v3 `model_25600` | ASAP fork, tag `dab-v6-recipe` (af60acf) |
| `finetune_dab_v5.sh` | v5: tighter tracking tolerances | v4 `model_41600` | same |
| `finetune_dab_v5_resume.sh` | v5 resumed after the run died | v5 `model_6700` | same |
| `finetune_dab_v6.sh` | v6: clean spawns, PPO safeguards (**deployed**, `model_20000`) | v5 resume `model_13000` | same |
| `config_v1_20260328_172005.yaml` | saved Hydra config of the v1 base run | | ASAP `logs/DabTracking/20260328_172005-DabTracking_wall_domainrand-...` |
| `config_v6_20260718_015831.yaml` | saved Hydra config of the v6 run | | ASAP `logs/DabTracking/20260718_015831-DabTracking_grounded_v6-...` |

The v1 base run had no script; its command is reconstructed in `docs/PIPELINE.md` (step 6 uses it). Each command was checked by composing it with Hydra and comparing with the saved config: they match except for `visualize_motion_file: null`, which came from an uncommitted local change to `base.yaml`.

`finetune_dab_v7.sh` (foot penalties, shelved because it lost push recovery) is on the fork's tag but deliberately not copied here.
