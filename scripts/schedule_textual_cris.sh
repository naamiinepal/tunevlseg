#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

echo "Running coop with CRIS"
bash -x scripts/schedule_coop_cris.sh

echo "Running cocoop with CRIS"
bash -x scripts/schedule_cocoop_cris.sh
