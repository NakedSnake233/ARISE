"""Replay selected CARE figures without overwriting the matched baseline tables."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    path = ROOT / 'Paper/fse2027_revision_20260926/figures/generate_review_figures.py'
    spec = importlib.util.spec_from_file_location('care_review_figures', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Missing retained figure generator')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STRICT_CANDIDATE = module.strict_candidate_analysis()
    module.operating_boundaries()
    module.fault_types()
    module.sensitivity()
    module.architecture_pdf()
    (path.parent / 'table_ablation.tex').write_text(module.ablation_table())
    print('Replayed operating-boundary, fault-type, sensitivity, architecture, and ablation figures/tables.')


if __name__ == '__main__':
    main()
