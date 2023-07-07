#!/usr/bin/env bash

JOB_TYPE=process_all 
DATA_FOL=data/all
MODES=(test_cd test_id valid)

for MODE in ${MODES[@]}; do
  echo "Preparing $MODE"
  if [[ ("$MODE" == "valid" || "$MODE" == "train") ]]; then
    JSON_FN=$DATA_FOL/train_period_data.jsonlist
  else
    JSON_FN=$DATA_FOL/heldout_period_data.jsonlist
  fi

  CMU_FN=data/cmu_ids/fixed_node_ids-$MODE.txt
  SAVE_FN=$DATA_FOL/$MODE-allthread_pairs.jsonlist #.unlabel

  python code/data_prep/cmv_prep.py --json_fn $JSON_FN --save_fn $SAVE_FN --cmu_fn $CMU_FN --job_type $JOB_TYPE 
done
