#!/bin/bash
# Re-run the EagleEye window sensitivity scan at higher statistics
# (300 null / 100 inj per grid point), two parallel lanes of n-jobs=4 (8 cores).
# Degenerate Repechage stats (T_rep/Lam_max) at small windows are flagged N/A
# in-script and persisted to the npz. Logs per window in logs/.
set -u
cd "$(dirname "$0")"
PY="$HOME/Documents/Environments/eagleeye/bin/python"
GRID="10,20,40,70,100,150,200"
NNULL=300
NINJ=100
NJOBS=4
mkdir -p logs
rm -f rescan_complete.flag

run_window () {
  local W=$1
  echo "[$(date '+%F %T')] START W=${W}s" >> logs/rescan_driver.log
  "$PY" -u run_sensitivity_window.py NuMix --window "$W" \
      --n-null "$NNULL" --n-inj-trials "$NINJ" --n-inj-grid "$GRID" \
      --n-jobs "$NJOBS" --seed 0 \
      > "logs/W${W}s.log" 2>&1
  echo "[$(date '+%F %T')] DONE  W=${W}s (exit $?)" >> logs/rescan_driver.log
}

# Lane A and Lane B run concurrently; windows within a lane run sequentially.
( run_window 200; run_window 50; run_window 10 ) &
LANE_A=$!
( run_window 100; run_window 20 ) &
LANE_B=$!

wait $LANE_A $LANE_B
echo "[$(date '+%F %T')] ALL WINDOWS DONE" >> logs/rescan_driver.log
touch rescan_complete.flag
