#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

echo "Running coop with CLIPSeg"
bash ./schedule_coop_clipseg.sh

echo "Running coop with CRIS"
bash ./schedule_coop_cris.sh