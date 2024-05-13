#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

echo "Running VPT with CLIPSeg"
bash ./schedule_vpt.sh

echo "Running Shared Separate with CLIPSeg"
bash ./schedule_separate.sh

echo "Running Shared Attention with CLIPSeg"
bash ./schedule_shared_atn.sh

echo "Running Maple with CLIPSeg"
bash ./schedule_maple.sh
