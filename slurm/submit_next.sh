#!/bin/bash
# Submit exactly ONE job: the next one of the Phase 0 plan (slurm/phase0_plan.tsv).
# Run by the user, by hand, once per job (cluster fair-use policy: max 1 running + 1 waiting, 12 h, manual).
# Refuses if 2 jobs exist or a job is already waiting. Details: src/submit_next.py.
#   slurm/submit_next.sh            submit the next job
#   slurm/submit_next.sh --dry-run  show what would be submitted, submit nothing
exec python3 /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/src/submit_next.py "$@"
