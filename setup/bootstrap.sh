#!/usr/bin/env bash
#
# Clone the three repositories this pipeline runs on, each at the exact commit
# pinned in forks.lock, into one folder (default: ./external).
#
#   bash setup/bootstrap.sh [TARGET_DIR]
#
# Then:
#   export ASAP_ROOT=$TARGET_DIR/ASAP RL_SAR_ROOT=$TARGET_DIR/rl_sar TRAM_ROOT=$TARGET_DIR/tram
#
# Existing clones are not re-cloned; they are fetched and checked out at the
# pinned commit, but only if their working tree is clean. Nothing is pushed.
# After this, follow setup/ASSETS.md for the files nobody may redistribute
# (Isaac Gym, SMPL, TRAM weights) and create the conda envs (setup/env_*.yml).

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lock="$here/forks.lock"
target="${1:-$here/external}"
mkdir -p "$target"

declare -A dir_for=([asap]=ASAP [rl_sar]=rl_sar [tram]=tram)

grep -Ev '^\s*(#|$)' "$lock" | while read -r name url ref commit; do
  dest="$target/${dir_for[$name]:-$name}"
  echo "== $name: $url @ $ref ($commit)"
  if [[ -d "$dest/.git" ]]; then
    if [[ -n "$(git -C "$dest" status --porcelain --untracked-files=no)" ]]; then
      echo "   $dest has uncommitted changes; leaving it alone" >&2
      exit 1
    fi
    git -C "$dest" fetch --quiet origin "$ref" || git -C "$dest" fetch --quiet origin
  else
    git clone --quiet "$url" "$dest"
    git -C "$dest" fetch --quiet origin "$ref" || true
  fi
  git -C "$dest" -c advice.detachedHead=false checkout --quiet "$commit"
  if [[ -f "$dest/.gitmodules" ]]; then
    git -C "$dest" submodule update --init --recursive --quiet
  fi
  got="$(git -C "$dest" rev-parse HEAD)"
  [[ "$got" == "$commit" ]] || { echo "   expected $commit, got $got" >&2; exit 1; }
  echo "   ok: $dest"
done

cat <<EOF

Done. Suggested environment:
  export ASAP_ROOT=$target/ASAP
  export RL_SAR_ROOT=$target/rl_sar
  export TRAM_ROOT=$target/tram
Next: setup/ASSETS.md, then the conda envs in setup/env_hvgym.yml and setup/env_tram.yml.
EOF
