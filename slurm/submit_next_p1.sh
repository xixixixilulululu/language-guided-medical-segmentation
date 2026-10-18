#!/bin/bash
# Submit exactly ONE Phase 1 job: the next one of slurm/phase1_plan.tsv, or the next segment of an unfinished
# run. Run by the user, by hand, once per job (cluster fair-use policy: max 1 running + 1 waiting, 12 h,
# manual). Refuses if 2 jobs exist or a job is already waiting. Details: src/submit_next_p1.py.
#   slurm/submit_next_p1.sh            submit the next job
#   slurm/submit_next_p1.sh --dry-run  show what would be submitted, submit nothing
exec python3 /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/src/submit_next_p1.py "$@"
