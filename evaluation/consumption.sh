#!/usr/bin/env bash

JOBRUN=0; JOBID=`sbatch evaluation/forecast.sh |awk '{print $(NF)}'`; while true; do ACCT=$(sacct -X --format ElapsedRaw -j ${JOBID} --noconvert -n -p); STAT=$(sstat --format ConsumedEnergy,JobID,NodeList -n -P -a -j ${JOBID}.batch --noconvert 2>/dev/null); if [ ! -z "$STAT" ]; then JOBRUN=1; echo $ACCT$STAT; else if [ "$JOBRUN" = "1" ]; then break; fi; fi; sleep 1; done > Job_profile.csv


