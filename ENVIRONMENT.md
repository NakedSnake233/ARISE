# Recorded environments and measurement limits

## Baseline adapters

- Classical execution and replay: Python 3.9.13, NumPy 1.21.5,
  scikit-learn 1.0.2, pytest 7.1.2.
- Neural adapter execution: Python 3.8.18, NumPy 1.24.3,
  scikit-learn 1.3.0, PyTorch 1.8.1+cu111, CPU execution.
- Seeds: 42--46 for TraceAnomaly, Eadro, MULAN, and CARE; deterministic rows run
  once; configuration selection uses training/validation data only.

## CARE measurements inherited by the paper

The retained CARE model environment used PyTorch 2.6.0+cu124, NumPy 1.26.4,
scikit-learn 1.7.2, pandas 2.3.3, and Drain3 0.9.11. TrainTicket base inference
used CUDA and AIOps base inference used CPU. The exact GPU and CPU identity was
not linked to the saved latency repetitions and is therefore reported as a
measurement limitation. Feature extraction, disk I/O, outer fusion, and final sorting were outside
the timed region.

## LLM endpoint

The no-data compatibility smoke requested and returned `gpt-5.6-luna`. All
1,989 study responses returned that identifier. The official OpenCode Go
Responses route does not accept our requested temperature, top-p, or seed
overrides, so all three are omitted. Retained provider bodies report temperature
1, top-p .98, and medium reasoning effort. The frozen transport uses
concurrency eight, a 60-second timeout, and three retries after the initial
attempt.

API timing sums successful HTTP-attempt wall times and excludes preparation and
retry/backoff. The complete paper campaign has 1,989 logical calls and 1,989
first-attempt successes. The provider reports 7,070,635 tokens. Applying the
official September 27, 2026 price table gives USD 1.555094 of usage-equivalent
value; the subscription's incremental monetary charge is unknown.
