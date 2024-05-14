#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

echo "Running VPT with CLIPSeg"
bash -x scripts/schedule_vpt.sh

echo "Running Shared Separate with CLIPSeg"
bash -x scrips/schedule_separate.sh

echo "Running Shared Attention with CLIPSeg"
bash -x scripts/schedule_shared_atn.sh

echo "Running Maple with CLIPSeg"
bash -x scripts/schedule_maple.sh
