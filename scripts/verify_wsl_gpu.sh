#!/usr/bin/env bash
set -euo pipefail

run_check() {
  local label="$1"
  shift
  echo
  echo "== ${label} =="
  "$@"
}

run_check "Python" python3 --version
run_check "GPU" nvidia-smi

echo
echo "== CUDA compiler =="
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc not found"
fi

echo
echo "== Python packages =="
python3 - <<'PY'
import importlib.util as u

for name in ["torch", "triton", "numpy"]:
    print(f"{name}: {'installed' if u.find_spec(name) else 'missing'}")
PY

echo
echo "Week 1 passes when nvidia-smi, nvcc --version, and a CUDA hello world all work."

