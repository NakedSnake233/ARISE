"""Author-approved metric redundancy variant around unchanged official CIRCA."""
from __future__ import annotations
import json
import numpy as np

from . import rcaeval_worker as worker

POLICY = worker.ROOT / "exp_revision/circa_rank_prune_policy.json"


def select_columns(values, columns, column_services, services):
    policy = json.loads(POLICY.read_text())
    values = np.asarray(values, dtype=float)
    valid = [i for i in range(values.shape[1]) if np.isfinite(values[:, i]).all() and np.unique(values[:, i]).size > 1]
    groups = [[i for i in valid if column_services[i] == service] for service in services]
    order = [group[position] for position in range(max([len(g) for g in groups], default=0))
        for group in groups if position < len(group)]
    limit = max(0, len(values) - 2)
    if valid and not limit: raise ValueError("fewer than three observed rows for Fisher-Z rank-prune")
    centered = values[:, valid] - values[:, valid].mean(0)
    unit = centered / np.linalg.norm(centered, axis=0) if valid else centered
    lookup = {index: position for position, index in enumerate(valid)}
    selected = []; redundant = []; capped = []
    threshold = policy["correlation_eigenvalue_ratio_min"]
    for index in order:
        if len(selected) >= limit:
            capped.append(index); continue
        candidate = [*selected, index]
        matrix = unit[:, [lookup[k] for k in candidate]]
        eigenvalues = np.linalg.eigvalsh(matrix.T @ matrix)
        if eigenvalues[0] > threshold * eigenvalues[-1]: selected.append(index)
        else: redundant.append(index)
    audit = dict(method="CIRCA+rank-prune", observed_rows=len(values), original_columns=len(columns),
        varying_complete_columns=len(valid), maximum_columns=limit, eigenvalue_ratio_min=threshold,
        selected_columns=[columns[i] for i in selected],
        removed_redundant_columns=[columns[i] for i in redundant],
        removed_sample_cap_columns=[columns[i] for i in capped],
        removed_invalid_constant_columns=[columns[i] for i in range(len(columns)) if i not in valid],
        selected_services=[column_services[i] for i in selected], values_changed=False, uses_labels=False)
    return selected, audit


def main():
    original_infer = worker.infer

    def infer(method, values, times, columns, column_services, services, inject_time, seed, dataset="generic"):
        if method != "CIRCA": raise ValueError("rank-prune is a separately named CIRCA variant only")
        selected, audit = select_columns(values, columns, column_services, services)
        scores, receipt = original_infer(method, values[:, selected], times, [columns[i] for i in selected],
            [column_services[i] for i in selected], services, inject_time, seed, dataset)
        if selected and (not receipt.get("metric_ranks") or not np.any(scores)):
            raise RuntimeError("CIRCA+rank-prune returned no ranked observed service; selection audit: " + json.dumps(audit))
        receipt["rank_prune"] = audit
        return scores, receipt

    worker.infer = infer
    try: worker.main()
    finally: worker.infer = original_infer


if __name__ == "__main__": main()
