#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONDONTWRITEBYTECODE=1
for t in test_contrast test_shared test_state test_processor; do
  python3 "tests/$t.py"
done
echo "ALL CPU TESTS PASSED"
