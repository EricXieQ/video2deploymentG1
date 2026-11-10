# Inputs used by the rehearsal that are not in this repository

The rehearsal (`docs/REHEARSAL_moving_arms.md`) ran from a clean clone of the public ASAP fork at tag `dab-v6-recipe`. That fork does not ship the files below. They were copied, read-only, from the author's local ASAP checkout into the clone, at the same relative paths.

All of them are **user-supplied, not in the repo**. To repeat the rehearsal, produce your own (setup/ASSETS.md, docs/PIPELINE.md steps 1 to 4) or obtain them from the author.

| File (relative to the ASAP checkout) | sha256 | Why it was needed |
|---|---|---|
| `humanoidverse/data/shape/g1_29dof_anneal_23dof/shape_optimized_v1.pkl` | `673cb4ed97882c9a0dce32f735fbbf29a79f6ac238004a655ef9defa523353a5` | the G1 SMPL shape fit used by retargeting; copied for completeness (the rehearsal did not re-retarget) |
| `humanoidverse/data/motions/raw_tairantestbed_smpl/Eric_moving_arms_amass.npz` | `6e000889c16d6bd9a190c52efcd0fc2682f2c4eefd5aa74baf0b23758b20f155` | the SMPL motion from TRAM; kept beside the pkl as its source |
| `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_Eric_moving_arms_amass.pkl` | `9ec40a145cd318395baeca4d67e9fb87eb7712f54a47ee40d2d3c460a62f7744` | the retargeted reference motion; input of step 5 |

Not copied: the SMPL model files (`humanoidverse/data/smpl/`). Steps 5 to 8 do not need them; only retargeting (step 4) does.

Generated during the rehearsal (also not in the repository; kept in the rehearsal hand-off folder, which is not published):

| File | sha256 |
|---|---|
| `0-motions_raw_tairantestbed_smpl_Eric_moving_arms_amass_grounded.pkl` (output of `capture/ground_reference.py`) | `105180624c6d63ca03213101074dd1a31df3e4152296fee101c5a230d74a3d4f` |

The Eric_moving_arms npz was also regenerated from the TRAM result with `capture/convert_tram_to_asap.py` and is byte-identical to the copied one (same sha256), which is the converter's regression check.
