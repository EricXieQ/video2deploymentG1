#!/usr/bin/env bash
#
# Fine-tune the dab policy with off-trajectory entry randomization.
# Run from the ASAP repo root with the hvgym conda env active:
#
#   conda activate hvgym
#   cd ~/Project/humanoid/ASAP
#   bash finetune_dab.sh
#
set -e

# IsaacGym's compiled .so dynamically links against libpython3.8.so.1.0,
# which lives in the conda env but isn't on LD_LIBRARY_PATH by default.
export LD_LIBRARY_PATH="/home/eric/miniconda3/envs/hvgym/lib:${LD_LIBRARY_PATH}"

python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +robot=g1/g1_29dof_anneal_23dof \
  +rewards=motion_tracking/reward_motion_tracking_dm_2real \
  +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
  +domain_rand=domain_rand_base \
  +terrain=terrain_locomotion_plane \
  num_envs=4096 \
  algo.config.num_learning_iterations=100000 \
  project_name=DabTracking \
  experiment_name=DabTracking_wall_offtraj_finetune_v3 \
  +checkpoint=logs/DabTracking/20260328_172005-DabTracking_wall_domainrand-motion_tracking-g1_29dof_anneal_23dof/model_93600.pt \
  robot.motion.motion_file=humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_wall_background_dab_amass.pkl \
  env.config.noise_to_initial_level=2 \
  env.config.init_noise_scale.root_rot=2 \
  env.config.termination_scales.termination_gravity_x=0.95 \
  env.config.termination_scales.termination_gravity_y=0.95 \
  env.config.termination.terminate_by_low_height=True \
  env.config.termination_scales.termination_min_base_height=0.3 \
  +env.config.off_trajectory_entry_fraction=0.2 \
  algo.config.actor_learning_rate=1.e-4 \
  algo.config.critic_learning_rate=2.e-4 \
  headless=True