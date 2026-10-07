# Reproduce the local experiment

This submission uses `Qwen/Qwen2.5-0.5B-Instruct` on an NVIDIA RTX 3050 Laptop GPU
(4 GB). It is a declared model substitution for the lab's default 4B model.
The model card is https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct .
The downloaded revision is `7ae557604adf67be50417f59c2c2f167def9a775`.

Use Python 3.12 and the exact installed packages in `requirements-lock.txt`.
On Windows PowerShell, from the repository root:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r submission/requirements-lock.txt
$env:PYTHONIOENCODING = 'utf-8'
$env:COMPUTE_TIER = 'LAPTOP'
$env:BASE_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
$env:MASK_MODE = 'assistant-only'
$env:EPOCHS = '2'
$env:EVAL_LIMIT = '0'
$env:HF_HOME = Join-Path $env:TEMP 'lab21-hf-cache'
.venv/Scripts/python.exe scripts/run_local.py nb1 nb2 nb3 nb4 nb5
.venv/Scripts/python.exe scripts/run_local.py verify
```

Use a fresh copy for a fresh experiment: NB2 overwrites the frozen baseline and
NB3 retrains `correct`; NB4 resumes existing adapters. Do not mix old adapters
with new baselines, model choices, masks, or datasets. To resume this experiment,
request only the unfinished stages of `scripts/run_local.py`.

The LAPTOP recipe sets batch size 1 and gradient accumulation 8. NB3 and NB4
read NB1's `suggested_max_length`, capped by the tier limit, giving 256 here.
Both explicitly use the same derived optimizer-step budget. Seed is 42;
generation is greedy, batch size 4, maximum 160 new target tokens / 96 regression
tokens. Evaluation uses all 50 target and 15 regression examples. The 25-item
validation split is retained but not used for checkpoint selection. The secret
holdout is not used for training, prompt tuning, or qualitative selection.

The original broad requirement files remain unchanged. The lock records the
working Windows environment, including bitsandbytes, which the original Linux-only
marker would omit. TorchAO is not needed by these LoRA/bitsandbytes runs and is not
installed. Package installation and model downloads require network access;
after caching, set `$env:HF_HUB_OFFLINE = '1'` to run locally without network.

Evidence lives in `results/`: complete predictions, baseline timestamp and eval
hashes, mask checks, metrics, training CSV, qualitative comparisons, and console
logs. Each trained adapter directory also contains `trainer_state.json` with
its measured loss history. `final_loss` in `runs.csv` is the Trainer's mean training
loss over the run, not the final logged batch loss.

Changes made for this submission preserve full predictions, verify the trainer's
mask, record actual completed steps and loss curves, derive training length from
NB1, and check model/eval identity before comparing with frozen baselines.
Seed 42 is explicitly applied immediately before each SFTTrainer creates its LoRA
weights, in addition to the Trainer's seed. The installed TRL constructs adapters
before the underlying Trainer sets its seed. An initial training attempt without
this fix was excluded and all four training runs were restarted before evaluation;
the original frozen baselines were retained. Only the corrected runs are reported.
The scoring functions, optimized prompt, evaluation data, and verdict thresholds
are unchanged. Colab notebooks are regenerated from their Python sources.

Git converted the shipped JSONL files from LF to CRLF on Windows. The raw file
hashes therefore differ from `data/checksums.json`, but all LF-normalized hashes
match the reference and the original Git blobs. `results/checksum_audit.json`
records both hashes. The verifier accepts this exact newline conversion only;
tests confirm that label and whitespace edits still fail. No data file or
reference checksum was rewritten. NB2/NB5 additionally compare the full raw
SHA-256 hashes from this particular frozen experiment.
