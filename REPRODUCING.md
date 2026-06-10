# Reproducing the Benchmarks

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management

## Installation

```bash
git clone https://github.com/Galvanized-Heart/orthobo.git
cd orthobo
uv sync
```

Create a `.env` file so Hydra can resolve the project root:

```bash
echo "PROJECT_ROOT=." > .env
```

## Running a single experiment

```bash
uv run scripts/run_benchmark.py \
  sampler=orthobo \
  benchmark=hartmann6 \
  seed=42
```

**Available options:**

| Args | Options |
|---|---|
| `sampler` | `orthobo`, `naive`, `vanilla` |
| `benchmark` | `hartmann6`, `ackley10`, `levy16` |
| `seed` | any integer |

Default experiment settings are in `configs/config.yaml`:

```yaml
experiment:
  n_trials: 200
  n_startup_trials: 10
  mc_budget: 64
```

Any of these can be overridden on the command line:

```bash
uv run scripts/run_benchmark.py \
  sampler=orthobo \
  benchmark=levy16 \
  seed=1 \
  experiment.n_trials=50 \
  experiment.n_startup_trials=10 \
  experiment.mc_budget=64
```

Results are saved as `results.json` in a timestamped Hydra output directory under `outputs/`. A per run regret plot is saved alongside it.

## Running the full benchmark sweep

```bash
bash scripts/run_experiments.sh
```

This runs all 45 combinations (3 samplers * 3 benchmarks * 5 seeds) sequentially using Hydra multirun. Results are saved under `multirun/`. Results are saved incrementally, so partial runs are not lost if interrupted.

## Aggregating results and generating plots

```bash
uv run scripts/aggregate_and_plot.py
```

This scans all output directories under the project root, deduplicates by `(sampler, benchmark, seed, n_trials, n_startup_trials, mc_budget)` keeping the most recent run per configuration, and saves one plot per benchmark to `figures/`.

If you run additional experiments after the initial sweep (e.g. to fill in missing seeds), just rerun the aggregation script, it handles deduplication automatically.