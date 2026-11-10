#!/usr/bin/env bash
#
# Stage 1: train a motion-tracking policy from scratch, with domain randomization.
#
# This is the dab's verified "v1 base" command (docs/PIPELINE.md, step 6; the
# saved config of run 20260328_172005-DabTracking_wall_domainrand composes to the
# same config), with three changes:
#   1. motion file, project and experiment name are variables;
#   2. the iteration count is set (the dab ran open-ended and was stopped by hand);
#   3. control-delay randomization is widened from 0-2 to 0-3 policy steps
#      (0-40 ms -> 0-60 ms) to cover the ~50 ms command delay measured on the
#      real G1. This is a deliberate deviation from the dab; see docs/PAPER_VS_THIS.md.
#
# Run from anywhere; it changes into $ASAP_ROOT. Use tmux: a dropped X session
# killed one dab run and another died without a log.
#
#   export ASAP_ROOT=~/code/ASAP            # the pinned fork (setup/bootstrap.sh)
#   MOTION_PKL=$ASAP_ROOT/humanoidverse/data/motions/.../my_motion_grounded.pkl \
#   PROJECT=MyMotion EXPERIMENT=MyMotion_stage1 ITERATIONS=20000 \
#     bash train/stage1_pretrain.sh
#
# Variables (defaults in brackets):
#   ASAP_ROOT     ASAP fork checkout                         [required]
#   MOTION_PKL    retargeted, grounded reference .pkl         [required]
#   PROJECT       logs/<PROJECT>/... folder name              [MotionTracking]
#   EXPERIMENT    run name suffix                             [<PROJECT>_stage1]
#   ITERATIONS    PPO iterations to run                       [20000]
#   CHECKPOINT    optional model_N.pt to resume from          [none: from scratch]
#   NUM_ENVS      parallel robots                             [4096]
#   CTRL_DELAY_MAX  max control delay in 20 ms policy steps   [3]
#   CONDA_ENV_PREFIX  hvgym env prefix, for LD_LIBRARY_PATH   [$CONDA_PREFIX]
#   PYTHON        python to use                               [$CONDA_ENV_PREFIX/bin/python]
#   LOG_FILE      console log (tee)                           [$ASAP_ROOT/logs/<PROJECT>/<EXPERIMENT>_<time>.log]
#   Extra Hydra overrides can be appended as script arguments.
#
# Every Isaac Gym process must run headless. This script unsets DISPLAY.

set -euo pipefail

: "${ASAP_ROOT:?set ASAP_ROOT to the ASAP fork checkout}"
: "${MOTION_PKL:?set MOTION_PKL to the reference motion .pkl}"
PROJECT="${PROJECT:-MotionTracking}"
EXPERIMENT="${EXPERIMENT:-${PROJECT}_stage1}"
ITERATIONS="${ITERATIONS:-20000}"
CHECKPOINT="${CHECKPOINT:-}"
NUM_ENVS="${NUM_ENVS:-4096}"
CTRL_DELAY_MAX="${CTRL_DELAY_MAX:-3}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${CONDA_PREFIX:-}}"
: "${CONDA_ENV_PREFIX:?activate the hvgym env or set CONDA_ENV_PREFIX}"
PYTHON="${PYTHON:-${CONDA_ENV_PREFIX}/bin/python}"

MOTION_PKL="$(realpath "$MOTION_PKL")"
[[ -f "$MOTION_PKL" ]] || { echo "no such motion file: $MOTION_PKL" >&2; exit 1; }
if [[ -n "$CHECKPOINT" ]]; then
  CHECKPOINT="$(realpath "$CHECKPOINT")"
  [[ -f "$CHECKPOINT" ]] || { echo "no such checkpoint: $CHECKPOINT" >&2; exit 1; }
fi

cd "$ASAP_ROOT"
# Isaac Gym's .so links against libpython3.8 from the conda env.
export LD_LIBRARY_PATH="${CONDA_ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# gymtorch is compiled on first import and needs the env's ninja on PATH, as
# `conda activate` would provide.
export PATH="${CONDA_ENV_PREFIX}/bin:${PATH}"
# Import humanoidverse from this checkout, not from whatever was pip-installed -e.
export PYTHONPATH="${ASAP_ROOT}:${PYTHONPATH:-}"
unset DISPLAY

where="$("$PYTHON" -c 'import humanoidverse, os; print(os.path.dirname(os.path.dirname(os.path.abspath(humanoidverse.__file__))))')"
if [[ "$(realpath "$where")" != "$(realpath "$ASAP_ROOT")" ]]; then
  echo "humanoidverse imports from $where, not $ASAP_ROOT; fix PYTHONPATH" >&2
  exit 1
fi

mkdir -p "logs/${PROJECT}"
LOG_FILE="${LOG_FILE:-$ASAP_ROOT/logs/${PROJECT}/${EXPERIMENT}_$(date +%Y%m%d_%H%M%S).log}"
echo "ASAP_ROOT=$ASAP_ROOT ($(git -C "$ASAP_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))"
echo "motion=$MOTION_PKL iterations=$ITERATIONS envs=$NUM_ENVS ctrl_delay=0..$CTRL_DELAY_MAX steps"
echo "log=$LOG_FILE"

ckpt_args=()
[[ -n "$CHECKPOINT" ]] && ckpt_args=("+checkpoint=$CHECKPOINT")

"$PYTHON" humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +domain_rand=domain_rand_base \
  +rewards=motion_tracking/reward_motion_tracking_dm_2real \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
  num_envs="$NUM_ENVS" \
  project_name="$PROJECT" \
  experiment_name="$EXPERIMENT" \
  robot.motion.motion_file="$MOTION_PKL" \
  rewards.reward_penalty_curriculum=True \
  rewards.reward_penalty_degree=0.00001 \
  env.config.resample_motion_when_training=False \
  env.config.termination.terminate_when_motion_far=True \
  env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
  env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
  env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
  robot.asset.self_collisions=0 \
  domain_rand.ctrl_delay_step_range="[0,${CTRL_DELAY_MAX}]" \
  algo.config.num_learning_iterations="$ITERATIONS" \
  headless=True \
  "${ckpt_args[@]}" \
  "$@" 2>&1 | tee "$LOG_FILE"
