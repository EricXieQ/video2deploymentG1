#!/usr/bin/env bash
#
# Stage 2: deployment fine-tune, warm-started from a stage-1 checkpoint.
#
# The dab reached its deployed policy through four cumulative fine-tunes
# (v3 off-trajectory entry, v4 ankle penalty, v5 tighter tracking, v6 clean
# spawns plus PPO safeguards; scripts in train/dab_as_run/). Each one only added
# or changed flags, so this script applies v6's final flag set in one run:
#
#   reward  motion_tracking/reward_motion_tracking_dm_2real_v5    (v4 ankle penalty + v5 tolerances)
#   +env.config.off_trajectory_entry_fraction=0.2   20% of episodes start at phase 0 from the walking stance (v3)
#   env.config.noise_to_initial_level=0.25          clean, upright spawns (v6; v3-v5 used 2)
#   tilt and low-height termination                  (v3)
#   actor / critic learning rate 1e-4 / 2e-4         (v3)
#   +algo.config.max_learning_rate=3.e-4             adaptive-LR cap (v6)
#   algo.config.max_grad_norm=0.5                    tighter gradient clip (v6)
#
# plus the same control-delay widening as stage 1 (0-3 policy steps = 0-60 ms).
# Motion-far termination stays off, as in v3-v6 (the exp default).
#
# Doing all of this in a single jump is NEW and was first tested on the
# Eric_moving_arms rehearsal (docs/REHEARSAL_moving_arms.md).
#
# Do not add foot-motion penalties here: the dab's v7 did, and lost push recovery.
#
#   export ASAP_ROOT=~/code/ASAP
#   MOTION_PKL=.../my_motion_grounded.pkl PROJECT=MyMotion EXPERIMENT=MyMotion_stage2 \
#   CHECKPOINT=$ASAP_ROOT/logs/MyMotion/<stage1-run>/model_20000.pt ITERATIONS=5000 \
#     bash train/stage2_deploy_finetune.sh
#
# Variables (defaults in brackets):
#   ASAP_ROOT, MOTION_PKL, CHECKPOINT            [required]
#   PROJECT [MotionTracking]  EXPERIMENT [<PROJECT>_stage2]  ITERATIONS [5000]
#   NUM_ENVS [4096]  CTRL_DELAY_MAX [3]  CONDA_ENV_PREFIX [$CONDA_PREFIX]
#   PYTHON [$CONDA_ENV_PREFIX/bin/python]  LOG_FILE [$ASAP_ROOT/logs/<PROJECT>/<EXPERIMENT>_<time>.log]
#   Extra Hydra overrides can be appended as script arguments.
#
# ITERATIONS counts from the checkpoint: resuming model_3000.pt with
# ITERATIONS=1500 ends at model_4500.pt.

set -euo pipefail

: "${ASAP_ROOT:?set ASAP_ROOT to the ASAP fork checkout}"
: "${MOTION_PKL:?set MOTION_PKL to the reference motion .pkl}"
: "${CHECKPOINT:?set CHECKPOINT to the stage-1 model_N.pt}"
PROJECT="${PROJECT:-MotionTracking}"
EXPERIMENT="${EXPERIMENT:-${PROJECT}_stage2}"
ITERATIONS="${ITERATIONS:-5000}"
NUM_ENVS="${NUM_ENVS:-4096}"
CTRL_DELAY_MAX="${CTRL_DELAY_MAX:-3}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${CONDA_PREFIX:-}}"
: "${CONDA_ENV_PREFIX:?activate the hvgym env or set CONDA_ENV_PREFIX}"
PYTHON="${PYTHON:-${CONDA_ENV_PREFIX}/bin/python}"

MOTION_PKL="$(realpath "$MOTION_PKL")"
CHECKPOINT="$(realpath "$CHECKPOINT")"
[[ -f "$MOTION_PKL" ]] || { echo "no such motion file: $MOTION_PKL" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "no such checkpoint: $CHECKPOINT" >&2; exit 1; }

cd "$ASAP_ROOT"
export LD_LIBRARY_PATH="${CONDA_ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${CONDA_ENV_PREFIX}/bin:${PATH}"
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
echo "motion=$MOTION_PKL checkpoint=$CHECKPOINT iterations=$ITERATIONS envs=$NUM_ENVS ctrl_delay=0..$CTRL_DELAY_MAX steps"
echo "log=$LOG_FILE"

"$PYTHON" humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +robot=g1/g1_29dof_anneal_23dof \
  +rewards=motion_tracking/reward_motion_tracking_dm_2real_v5 \
  +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
  +domain_rand=domain_rand_base \
  +terrain=terrain_locomotion_plane \
  num_envs="$NUM_ENVS" \
  algo.config.num_learning_iterations="$ITERATIONS" \
  project_name="$PROJECT" \
  experiment_name="$EXPERIMENT" \
  +checkpoint="$CHECKPOINT" \
  robot.motion.motion_file="$MOTION_PKL" \
  env.config.noise_to_initial_level=0.25 \
  env.config.init_noise_scale.root_rot=2 \
  env.config.termination_scales.termination_gravity_x=0.95 \
  env.config.termination_scales.termination_gravity_y=0.95 \
  env.config.termination.terminate_by_low_height=True \
  env.config.termination_scales.termination_min_base_height=0.3 \
  +env.config.off_trajectory_entry_fraction=0.2 \
  algo.config.actor_learning_rate=1.e-4 \
  algo.config.critic_learning_rate=2.e-4 \
  +algo.config.max_learning_rate=3.e-4 \
  algo.config.max_grad_norm=0.5 \
  domain_rand.ctrl_delay_step_range="[0,${CTRL_DELAY_MAX}]" \
  headless=True \
  "$@" 2>&1 | tee "$LOG_FILE"
