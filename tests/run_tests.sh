#!/usr/bin/env bash
#
# Fast, CPU-only checks for this repository. No GPU, no Isaac Gym.
#
#   bash tests/run_tests.sh
#
# Needs a Python with numpy, scipy, joblib, pyyaml, mujoco and torch: the `hvgym`
# env from setup/env_hvgym.yml. It uses, in order: $PYTHON, $CONDA_PREFIX/bin/python
# when that env is active, ~/miniconda3/envs/hvgym/bin/python, then python3.
#
# The reference-motion tests need the G1 robot model from the pinned ASAP fork.
# They use $ASAP_ROOT, else external/ASAP (setup/bootstrap.sh), else a sparse
# fetch of only the robot folder and robot config at the commit in forks.lock,
# cached under ${XDG_CACHE_HOME:-~/.cache}/video2deploymentG1/. If none of these
# is available (offline), those tests are skipped with the reason printed.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

pick_python() {
  local c
  for c in "${PYTHON:-}" "${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}" "$HOME/miniconda3/envs/hvgym/bin/python" python3; do
    [[ -n "$c" ]] || continue
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import numpy, scipy, joblib, yaml, mujoco, torch' >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  return 1
}

PY="$(pick_python)" || { echo "FAIL: no Python with numpy, scipy, joblib, pyyaml, mujoco and torch (see setup/env_hvgym.yml)" >&2; exit 1; }
# torch and mujoco wheels in a conda env link against the env's libraries.
py_prefix="$(dirname "$(dirname "$(command -v "$PY")")")"
export LD_LIBRARY_PATH="${py_prefix}/lib:${LD_LIBRARY_PATH:-}"
unset DISPLAY
echo "python: $PY"

echo "== shell syntax"
while IFS= read -r f; do
  bash -n "$f" && echo "ok  $f"
done < <(git ls-files '*.sh')

echo "== python byte-compile"
tmp_pyc="$(mktemp -d)"
trap 'rm -rf "$tmp_pyc"' EXIT
while IFS= read -r f; do
  PYTHONPYCACHEPREFIX="$tmp_pyc" "$PY" -m py_compile "$f" && echo "ok  $f"
done < <(git ls-files '*.py')

echo "== pinned ASAP robot model"
if [[ -z "${ASAP_ROOT:-}" ]]; then
  if [[ -d external/ASAP/humanoidverse ]]; then
    ASAP_ROOT="$here/external/ASAP"
  else
    read -r _ asap_url _ asap_commit < <(grep -E '^asap[[:space:]]' forks.lock)
    cache="${XDG_CACHE_HOME:-$HOME/.cache}/video2deploymentG1/asap-${asap_commit}"
    if [[ ! -f "$cache/.complete" ]]; then
      rm -rf "$cache"
      if git init -q "$cache" \
         && git -C "$cache" remote add origin "$asap_url" \
         && git -C "$cache" sparse-checkout set --no-cone \
              /humanoidverse/data/robots/g1/ /humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml \
         && git -C "$cache" fetch -q --depth 1 --filter=blob:none origin "$asap_commit" \
         && git -C "$cache" -c advice.detachedHead=false checkout -q FETCH_HEAD; then
        touch "$cache/.complete"
      else
        echo "skip: could not fetch the ASAP robot model from $asap_url ($asap_commit)"
        rm -rf "$cache"
      fi
    fi
    [[ -f "$cache/.complete" ]] && ASAP_ROOT="$cache"
  fi
fi
if [[ -n "${ASAP_ROOT:-}" ]]; then
  export ASAP_ROOT
  echo "ASAP_ROOT=$ASAP_ROOT"
else
  echo "ASAP_ROOT unset: reference-motion tests will be skipped"
fi

echo "== unit tests"
PYTHONDONTWRITEBYTECODE=1 "$PY" -m unittest discover -s tests -p 'test_*.py' -v
