#!/usr/bin/env bash
# Fetch the exact official PersonaMem-v3 evaluator used by the native protocol.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REVISION="fcbc03ec3514cca7db32e81c3a844d3875659404"
TARGET="${PM3_EVALUATOR_DIR:-$PROJECT_ROOT/data/raw/PersonaMem-v3-evaluator}"

if [ -e "$TARGET" ] && [ ! -d "$TARGET/.git" ]; then
  echo "Refusing to replace non-git path: $TARGET" >&2
  exit 2
fi

if [ ! -d "$TARGET/.git" ]; then
  git clone --filter=blob:none https://github.com/bowen-upenn/PersonaMem-v3.git "$TARGET"
fi

if [ -n "$(git -C "$TARGET" status --porcelain)" ]; then
  echo "Refusing to change dirty PM3 evaluator checkout: $TARGET" >&2
  exit 2
fi

git -C "$TARGET" fetch --depth 1 origin "$REVISION"
git -C "$TARGET" checkout --detach "$REVISION"
printf 'PM3 evaluator ready: %s @ %s\n' "$TARGET" "$REVISION"
