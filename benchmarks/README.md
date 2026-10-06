# Benchmarks

`bench.py` measures the parts of DemandCast whose cost matters when the pipeline runs nightly
over hundreds of store x SKU series. It depends on nothing beyond the package itself and prints
a JSON report.

```bash
python benchmarks/bench.py                      # full report (~20-40 s on 2 vCPUs)
python benchmarks/bench.py --quick              # fewer repetitions and series (what CI runs)
python benchmarks/bench.py --out benchmarks/results/bench.json   # also write the report
make bench                                      # same as the previous line
```

## What is measured

| Section | Measurement | Why it matters |
|---|---|---|
| `fit_predict` | median wall time of `fit` + 28-day `predict` per registered model on one 730-day series (`MODEL_REGISTRY`, plus the `promo_*` wrappers when `demandcast.promo` is present); Croston is timed on an intermittent series | every candidate is fitted once per backtest fold per series |
| `holt_winters` | the current `HoltWinters.fit` vs. a retained **scalar reference** (the v0.3.0 per-candidate recursion copied verbatim into `bench.py`) with the same 12-point grid; reports `speedup` and `sse_match` | Holt-Winters dominated the baseline run time; the vectorised grid search must stay numerically identical |
| `select_model` | full rolling-origin selection (4 folds, 28-day horizon) per series over 20 synthetic series (5 with `--quick`), plus the winner histogram | the per-series unit of work that `demandcast run` fans out over the process pool |
| `full_run` | `simulate.generate` of 3 stores x 8 products x 300 days into an in-memory database, then `pipeline.run` with 2 workers (`--workers`) | the end-to-end shape used by the test fixture and the CI smoke run |

The benchmark deliberately never builds the default 10 x 40 x 730 dataset; time it with the CLI
(`demandcast init`, `demandcast run --workers 2`) when you need the headline number.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--quick` | off | 3 repetitions, 5 series (instead of 10 / 20) |
| `--n DAYS` | 730 | synthetic series length |
| `--series K` | 20 (quick: 5) | series for the selection benchmark |
| `--reps R` | 10 (quick: 3) | timing repetitions per model |
| `--workers W` | 2 | process-pool size for the full run |
| `--seed S` | 42 | RNG seed for the synthetic series and the dataset |
| `--out PATH` | - | write the JSON report to a file as well |

## Reading the numbers

* Compare runs on the **same machine** only; the report records Python, NumPy and CPU count.
* `holt_winters.speedup` is the ratio of the scalar reference's median to the current
  implementation's median; `sse_match` must stay `true` (same best candidate and SSE).
* `select_model.mean_s_per_series x number of series / workers` approximates the run time of
  `demandcast run` before the write-back.
* `benchmarks/results/` is git-ignored: published figures belong in the README together with the
  machine they were measured on, never as bare numbers.
