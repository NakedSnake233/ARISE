# CARE — anonymous source snapshot

This repository contains the current CARE source, model code, experiment
runners, configurations, split definitions, and existing automated tests.
It is prepared for double-anonymous review. Share the anonymous mirror URL
with reviewers; the backing GitHub account is not an anonymous identity.

## Contents

- `scripts/care/`: data preparation, training, evidence correction, baselines,
  LLM transport, offline replay, aggregation, and table generation.
- `codes/`: multimodal localizer, preprocessing, and legacy diagnostic tools.
- `exp_revision/`: subsequent experiment and evaluation implementations and tests.
- `configs/care/`, `splits/`, `manifests/`: configurations and cohort definitions.
- `test/`: CARE and FSE baseline regression tests.

This is a source release. Raw telemetry, fitted models, experiment outputs,
API response caches, and manuscript files are not included. Consequently,
commands that need those inputs require data acquisition or a separate artifact
package first. No new experimental results or submission-readiness claims are
made by this release. Earlier and diagnostic method versions are retained as
source; their presence does not make them valid evidence for the paper.

## Environment and checks

Python 3.10 is recommended for CARE. Install `requirements-care.txt` in a
separate environment with a suitable PyTorch build for your hardware:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-care.txt
python -m pytest -q test/test_care_correction.py test/test_care_reproduction.py \
  test/test_care_review_experiments.py test/test_fse2027_baselines.py \
  test/test_fse2027_opencode_transport.py
```

`requirements.txt` records the older DGL localizer environment. The replay and
neural-recorded requirements describe earlier artifact environments. These are
alternative environments; do not install all requirement files together.
`ENVIRONMENT.md` is historical environment documentation, not a fresh-run log.

## Data and execution

See `DATA_ACQUISITION.md` and `manifests/cohort_protocol.json` for upstream data
and cohort information. Relevant commands expose their options through `--help`:

```bash
python scripts/care/prepare_aiops.py --help
python scripts/care/prepare_trainticket.py --help
python scripts/care/train_base.py --help
python scripts/care/run_review_experiments.py --help
```

`test/test_care_aiops_lora_v2.py` and some `exp_revision/tests/` checks
require prepared data, fitted models, or retained experiment outputs that are
not bundled in this source release. Run them after supplying those inputs.

Set data and output locations to paths on your own machine. `/workspace/CARE`
and `/workspace` in legacy defaults are anonymized example locations.
LLM model weights and LLaMA-Factory are external dependencies. Live LLM calls
require a runtime credential; offline tests do not require a credential.
Legacy Neo4j tools read `NEO4J_PASSWORD` from the environment.

## Provenance and anonymity

The snapshot includes uncommitted source changes from the working tree. It
contains no development history, workstation logs, agent configuration, SSH
material, or stored credentials. `ANONYMIZATION.json` lists source sanitization
changes. Only literal credentials and workstation identity/default paths were
changed; upstream names and source attribution are preserved.

Historical hash checks inside experiment configurations refer to their original
source files. Anonymized files have different hashes. Those historical receipts
must not be interpreted as verification of this release; establish a new run
manifest for fresh experiments. `SHA256SUMS` verifies this release's file bytes.

Dataset and third-party source terms continue to apply; see
`ARTIFACT_LICENSES.md`. This repository does not confer an upstream license.
The `codes/` localizer originates from Eadro; third-party source links in code
and protocol builders are retained for attribution.
