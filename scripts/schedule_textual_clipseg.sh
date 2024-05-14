#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

echo "Running coop and COCOOP with CLIPSeg"
bash scripts/schedule_coop_clipseg.sh

echo "Running COCOOP with CLIPSeg"
bash scripts/schedule_cocoop_clipseg.sh
