# Project rules (apply to every session)

## Language
- Every file pushed to GitHub (code, comments, notebooks, report `.md` files) is written in **English**.
  Before every push, check the staged files for Chinese characters, e.g.
  `git diff --cached --name-only | xargs grep -nP '[\x{4e00}-\x{9fff}]'` must print nothing.
- Chinese reading copies are named `*_zh_CN.md`, stay local only, and are excluded by `.gitignore`.

## Environment
- Home directory: `/mnt/cv_data/users/hahaxixi`.
- Conda environment `~/envs/lgms`; Jupyter kernel `lgms`.
- Before installing any package, do a dry run first (`pip install --dry-run ...`). Never change torch
  (`2.14.0+cu130`); a package whose install would upgrade, downgrade or replace torch is not installed.

## Compute
- Training runs are submitted as `.slurm` files to the Slurm partition `gpu`. Never train on the login node.

## Data and large files
- Datasets live in `~/datasets/`. Raw data, checkpoints and pretrained weights are never committed to git.

## Code style
- Code is commented.
- In notebooks, every code cell is preceded by a Markdown cell explaining it.

## Experimental integrity
- Checkpoints are selected on the validation set only. The test set is never used to tune settings or
  choose models.
- Every number in a report comes from an actual run. Numbers that have not been produced yet are left as
  placeholders.

## Figures that support a conclusion
- Use real cases. In one figure, side by side: original image + ground-truth mask (yellow dashed contour)
  + each model's prediction overlaid; every panel is labelled with that case's Dice and HD95.
- Cases are selected by a stated rule (e.g. Dice quantiles), never hand-picked.
- One or two sentences above the figure state the conclusion and point to the specific cases visible in it.

## Workflow
- For large changes, list the plan first and wait for confirmation before doing it.
- When work is done: commit and push, then report the commit hash.

## Phone notifications (ntfy)
- Every Slurm job must send an ntfy notification when it ends: success (with the best validation Dice
  where there is one), failure, or cancellation, always with the Slurm job ID. This is done by
  `slurm/notify_trap.sh`, sourced by `slurm/env.sh`; every new `.slurm` file sources `env.sh` and
  sets `JOB_LABEL` (and `NOTIFY_RESULT` where there is a result).
- Claude must send a notification whenever it finishes a task or stops to wait for confirmation.
  The project hooks in `.claude/settings.json` (`Stop`, `Notification`) do this via
  `src/claude_notify_hook.sh`. Before stopping, Claude writes a one-line status to
  `~/.cache/lgms/claude_status` (e.g. "Waiting for your confirmation: quick checks done"); the Stop
  hook sends it.
- Notifications go through `src/notify.sh`. The channel name is in `~/.ntfy_topic`; never write it
  into the repository. Messages contain only a task name, a status and a short result — no data,
  file contents, paths or credentials.

## Reference
- Full Phase 0 requirements: `reports/PHASE0_PLAN.md`.
