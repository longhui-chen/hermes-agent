#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
list="$root/scripts/test-harness/invariant-gated-packages.txt"
python_bin="${HERMES_PYTHON:-python3}"

while IFS= read -r path || [ -n "$path" ]; do
  path="${path%%#*}"
  path="$(printf '%s' "$path" | xargs)"
  [ -z "$path" ] && continue
  test -f "$root/$path"
  if ! grep -qE '^def test_random_' "$root/$path"; then
    echo "invariant gate: $path has no test_random_* sequence" >&2
    exit 1
  fi
  "$python_bin" -m pytest -q -p no:cacheprovider "$root/$path"
done < "$list"
