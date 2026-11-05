# AGENTS.md

This runbook is for coding agents that follow the prompts in the README's quickstart. The README is the reference for every command, and this file adds the order of the steps, the checks between them and the points where the user decides.

| Prompt | Steps |
|---|---|
| set up TAPESTRY, train the LIBERO recipe and evaluate it | 1 to 7 |
| check on the run | 7 |
| fine-tune the pretrained checkpoint on my robot demos | 1, 3 and 9 |
| run the pretrained policy on my robot | 1, 3 and 10 |

## Rules

- Each shell call may start in a fresh shell. Once the environment exists, start every call with `source <conda base>/etc/profile.d/conda.sh && conda activate tapestry` from the clone root, where `<conda base>` is what `conda info --base` prints.
- Run the README commands as written and pass any change as a command-line override. Do not edit code or configs, apart from the action-space entry in step 9 when the user approves it.
- When a command fails, stop. Apply a fix only if the README's Notes describe it, and otherwise show the user the error.
- Ask the user before downloading more than 10 GB, starting a GPU job that takes more than an hour, deleting or overwriting files, or installing anything outside the `tapestry` environment. The user's go-ahead for a plan covers the steps the plan lists.
- Installs and downloads can outlast an agent's default command timeout, so give them a long timeout or run them in the background and wait for them. Training and evaluation always run as background jobs (step 5), which outlive the agent session.
- Report the numbers you measured and never predict a success rate.

## 1. Check the machine

``` bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv
df -h . "$HOME" "${HF_HOME:-$HOME}"
conda --version; conda env list
command -v sbatch || echo "no sbatch"
```

If `conda` is not on the PATH, look for it in `~/miniconda3`, `~/miniforge3` and `~/anaconda3`, and ask the user before installing it. If an environment named `tapestry` already exists, ask the user whether to reuse it, because `conda create` would replace it. If `sbatch` exists, ask the user whether the jobs should run through Slurm (step 8). A missing `nvidia-smi` means that this machine has no GPU, and if `nvidia-smi` prints an error instead, stop and show it to the user.

## 2. Plan the run

Pick the run from the GPUs that `nvidia-smi` lists, whose memory is given in MiB:

| GPUs | Run |
|---|---|
| 4 or more with at least 140000 MiB | the LIBERO recipe as written |
| 8 or more with at least 80000 MiB | the LIBERO recipe with `--num_processes 8 training.batch_size=16`, which we have not run and which may run out of memory, so tell the user |
| 1 or more with at least 80000 MiB | the overfit check |
| 1 or more with at least 32000 MiB | the install check and the LoRA smoke test |
| anything smaller | the install check |

The install check is steps 3 and 4. It needs a GPU for the simulator, so on a machine without one it stops after the tests in step 3. On a machine that fits neither training run, tell the user that the recipe ran on four 141 GB GPUs and that the overfit check needs one GPU with at least 80 GB.

Then tell the user the plan and wait for one go-ahead. The plan names the run and the GPUs it will use (prefer GPUs whose `memory.used` is near zero), the downloads, the disk the run needs next to the free space that `df` reported, and how long it takes. The downloads are the LIBERO demonstrations (all five suites at about 100 GB for the recipe, or `libero_object` alone at about 7 GB) and, for any training run, the base models at about 49 GB in the HuggingFace cache (`$HF_HOME`, by default `~/.cache/huggingface`). The environment takes about 7 GB, and the table in the README's LIBERO section gives the disk and time of the recipe and the overfit check.

## 3. Build the environment

Run the first code block of the README's Setup from the clone root, leaving out its `git clone` and `cd` lines since you are already in the clone. When the task involves LIBERO (steps 4 to 6), run the first code block of the README's "Install LIBERO" as well, since steps 4 and 6 download the demonstrations. Three additions:

- If `nvidia-smi` reports a driver older than 580, run the README's `pip install torch==2.7.0 ...` line before `pip install -e .`.
- Before installing LIBERO, run `conda env config vars set LIBERO_CONFIG_PATH="$PWD/LIBERO/.libero"` and activate the environment again. LIBERO then keeps its config inside the clone at `$LIBERO_CONFIG_PATH/config.yaml`, which takes the place of `~/.libero/config.yaml` in the README, and leaves the config of any other LIBERO install untouched.
- Also before installing LIBERO, run `cmake --version` and `c++ --version` in the environment. If either fails, install both with the conda line under "Issues with `egl_probe`" in the README's Notes.

LIBERO's `pip install -r requirements.txt` ends with a pip `ERROR` that lists dependency conflicts, for example with transformers and numpy. This is expected, because the last line of that block reinstalls our versions. Then check the install:

``` bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
pip install -e ".[test]" && CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python -m pytest tests/ -q
cat "${HF_HOME:-$HOME/.cache/huggingface}/accelerate/default_config.yaml" 2>/dev/null || echo "no accelerate default config"
```

The tests run on the CPU and download nothing. `accelerate launch` applies a default config to every run, so if one exists, show it to the user and ask whether to keep it or move it aside.

## 4. Check the simulator

``` bash
hf download yifengzhu-hf/LIBERO-datasets --repo-type dataset --local-dir LIBERO/libero/datasets --include "libero_object/*"
mkdir -p preprocessed_data logs && ln -s "$(realpath LIBERO/libero/datasets)" preprocessed_data/libero_demos
python scripts/verify_libero_data.py --demo-dir preprocessed_data/libero_demos --suite libero_object --task 0 \
    --demos 10 --output logs/replay.json
```

This replays the recorded actions of 10 demonstrations in the simulator, independently of any model. The last line reads `Demo replay: K/10 actual simulator successes; NOT policy success`, and we got 10/10 with the simulator versions in the README. A lower count means that the simulator behaves differently from ours, so stop and report it.

## 5. Run a background job

Each job gets a short name (`recipe`, `overfit`, `smoke` or `finetune`). Write its commands from step 6 into a script and start it detached:

``` bash
mkdir -p logs
cat > logs/NAME.sh <<'EOF'
cd <clone root>
source <conda base>/etc/profile.d/conda.sh
conda activate tapestry
set -e
echo $$ > logs/NAME.pid
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=<the GPUs from the plan>
<the job's commands>
echo "JOB DONE"
EOF
setsid nohup bash logs/NAME.sh > logs/NAME.log 2>&1 < /dev/null &
```

`kill -0 $(cat logs/NAME.pid)` succeeds while the job runs, and `kill -- -$(cat logs/NAME.pid)` stops it together with all of its processes. A job that ended without printing `JOB DONE` failed, so show the user the end of its log. Before your session ends, tell the user where the log is and that the README's prompt for checking on a run reports its progress.

## 6. Jobs

The training log names the run's checkpoint directory, which is `checkpoints/<experiment_name>` or, if that exists from an earlier run, `checkpoints/<experiment_name>_<n>`. Each job reads it into `ckpt` after training.

**LIBERO recipe** (`recipe`):

``` bash
hf download yifengzhu-hf/LIBERO-datasets --repo-type dataset --local-dir LIBERO/libero/datasets
accelerate launch --multi_gpu --num_processes 4 train.py --config-name train_libero
ckpt=$(grep -a -o -m1 "Saving checkpoints to [^ ]*" logs/recipe.log | cut -d' ' -f4)
python eval_libero_parallel.py checkpoint=$ckpt/epoch_7.pt \
    'suites=[libero_10,libero_object,libero_spatial,libero_goal]' n_rollouts_per_task=50 \
    results_file=logs/recipe_eval.json
```

The training log also prints in-training results on the four suites (`LIBERO evaluation results`) after epochs 3 and 6.

**Overfit check** (`overfit`). First link the demo file with the `mkdir -p` and `ln -s` lines of the README's "Overfit one LIBERO task", then run:

``` bash
accelerate launch --num_processes 1 train.py --config-name overfit_libero training.dataset.root_dir=preprocessed_data/overfit
ckpt=$(grep -a -o -m1 "Saving checkpoints to [^ ]*" logs/overfit.log | cut -d' ' -f4)
for e in 15 20; do
    python eval_libero_parallel.py checkpoint=$ckpt/epoch_$e.pt 'suites=[libero_object]' 'task_ids=[0]' \
        n_rollouts_per_task=50 n_envs=5 results_file=logs/overfit_eval_epoch$e.json
done
```

**LoRA smoke test** (`smoke`). Link the demo file as for the overfit check, then run:

``` bash
accelerate launch --num_processes 1 train.py --config-name overfit_libero training.dataset.root_dir=preprocessed_data/overfit \
    experiment_name=libero_smoke quick=true training.num_epochs=1 training.batch_size=2 \
    decoders.cogvideox.use_lora=true decoders.cogvideox.freeze_pretrained_weights=true
ckpt=$(grep -a -o -m1 "Saving checkpoints to [^ ]*" logs/smoke.log | cut -d' ' -f4)
python eval_libero_parallel.py checkpoint=$ckpt/latest.pt 'suites=[libero_object]' 'task_ids=[0]' \
    n_rollouts_per_task=5 n_envs=5 results_file=logs/smoke_eval.json
```

The smoke test trains for a few steps on 10 samples, so it shows that training, checkpointing and evaluation run on this machine and says nothing about task success.

## 7. Check on a run

``` bash
ls -t logs/*.pid
kill -0 $(cat logs/NAME.pid) 2>/dev/null && echo running || echo ended
tr '\r' '\n' < logs/NAME.log | grep -a -E "^Epoch [0-9]+/|Saved checkpoint to|successes$|Error|JOB DONE" | tail -n 20
tail -c 2000 logs/NAME.log | tr '\r' '\n' | tail -n 3
```

The third command lists the finished epochs, the saved checkpoints, the result of each evaluated task, errors and the end of the job. It leaves out the `LIBERO evaluation results` summaries, because the summary of a final evaluation lists every episode on one line. The last command shows the live progress bar. The bar counts epochs from 0, so it reads `Epoch 0` during the first epoch, and the `Epoch 1/7` line marks the end of that epoch. The evaluator rewrites the job's `results_file` (step 6) after every finished task, and this prints the successes so far:

``` bash
python - <results file> <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
for suite, tasks in r.items():
    if isinstance(tasks, dict) and suite != "eval_settings":
        done = [t for t in tasks.values() if isinstance(t, dict)]
        k, n = sum(t["successes"] for t in done), sum(t["num_rollouts"] for t in done)
        print(f"{suite}: {k}/{n} successes over {len(done)} tasks")
if "overall_success_rate" in r:
    print("overall success rate:", r["overall_success_rate"])
EOF
```

The overall success rate appears once the evaluation has finished.

## 8. Slurm clusters

Ask the user for the partition, account, GPU type and count per node, GPU request syntax and time limit. Plan the run of step 2 from those GPUs, since `nvidia-smi` on the login node does not show them, and run the torch check of step 3 and the replay of step 4 on a GPU node through `srun`. Start the step 5 script with `#!/bin/bash`, the `#SBATCH` lines for one node, `#SBATCH --output=logs/NAME.log` and `#SBATCH --open-mode=append`, and leave `CUDA_VISIBLE_DEVICES` out of its `export` line, because Slurm sets it. Submit it with `sbatch` in place of the `setsid nohup` line. `squeue -u $USER` then replaces the `kill -0` check and `scancel` replaces `kill`.

A run longer than the time limit continues in chained jobs. Each next job runs a copy of the script whose training command adds `resume=true`, which continues the newest run of that experiment from its `latest.pt`, and for `finetune` drops `init_from=...`, which cannot be combined with `resume=true`. Submit it with `--dependency=afternotok:<the ids of all earlier jobs, joined by colons>`, so that it starts only after every earlier job stopped without finishing the run. Training overwrites `latest.pt` every 500 steps (`training.save_interval`), and a resumed run continues its unfinished epoch after the batches that `latest.pt` recorded, so a job loses only the steps after its last save. After a job prints `JOB DONE`, the jobs still queued can never start, so cancel them with `scancel`.

## 9. Fine-tune on the user's robot

Ask the user for the directory with their demos, the action space and the cameras the policy should see, in order. Then check one demo against the table in step 1 of the README's fine-tuning section:

``` bash
python -c 'import h5py, sys; h5py.File(sys.argv[1], "r").visititems(lambda n, o: print(n, getattr(o, "shape", ""), getattr(o, "dtype", "")))' <demo>.hdf5
```

- Demos in another format need converting to this layout. Offer to write a converter that writes new files and leaves the originals unchanged.
- A missing `text` dataset is added with `scripts/add_text_to_hdf5.py` once the user gives the task description and agrees, since the script writes into their files.
- The directory needs at least 6 demos, because the last 10% are held out for validation.
- An action space outside the built-in ones needs a new entry in `ACTION_SPACE_CONFIGS`, which you add only after the user approves its definition. For any action space other than `eepose_6drot`, leave out `--relative-rotation-actions` and add the overrides from the README's first bullet under fine-tuning step 3.

Then tell the user the plan and wait for one go-ahead, as in step 2. The run needs one GPU with at least 32 GB, downloads the 7.6 GB checkpoint and the base models, and writes the checkpoints that fine-tuning step 3 lists. Download `tapestry.pt` with the `hf download` line under the README's "Pretrained checkpoint". Compute the statistics as in fine-tuning step 2, then run steps 3 and 4 as a background job named `finetune` on that GPU:

``` bash
accelerate launch --num_processes 1 train.py --config-name finetune init_from=tapestry.pt <the overrides of step 3>
ckpt=$(grep -a -o -m1 "Saving checkpoints to [^ ]*" logs/finetune.log | cut -d' ' -f4)
python scripts/prepare_release_checkpoint.py --checkpoint $ckpt/epoch_50.pt --output my_robot.pt --verify
```

After the first epoch, tell the user how long an epoch takes, so they know when the 50 epochs finish.

## 10. Run a policy on the user's robot

The pretrained checkpoint drives our bimanual robot through our three cameras, so on any other robot fine-tune it first (step 9) and run the exported `my_robot.pt`. Inference needs about 10 GB of GPU memory with `text_cache_dir` set as in the README, or 28 GB without caching text prompts, and loading the checkpoint downloads the base models of step 2 unless they are cached.

1. Load the checkpoint as in the README's "Run a policy" section and print `policy.cameras`, `policy.image_size`, `policy.action_horizon` and `policy.action_dim`.
2. Before the robot moves, run `act` on the camera frames of a recorded demo at several steps `t`, with the demo's `actions[t]` as `ref_action`, and compare each chunk with `actions[t:t+action_horizon]`, as training pairs them. Show the user the error.
3. Write the control loop after `scripts/policy_example.py`. Resize each frame to `image_size` without cropping, pass the cameras in `policy.cameras` order with the current pose as `ref_action`, and execute part of each chunk before re-planning as the README describes.
4. Send no command to the robot until the user says to, and start with the user ready to stop it.

## Working on the code

The package lives in `tapestry/` (models, losses, data loaders and LIBERO evaluation) and the Hydra configs in `cfg/`, and the entry points are `train.py`, `eval_libero_parallel.py` and the scripts in `scripts/`. `CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python -m pytest tests/ -q` runs the tests on the CPU in under a minute. The recipe configs reproduce the paper, so leave them unchanged and pass experiments as overrides.
