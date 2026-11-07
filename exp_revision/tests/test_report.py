import numpy as np
import pandas as pd
from exp_revision import report as r


def rows(dataset="toy", scenario="S0"):
    result = []
    for event in range(12):
        for seed in (42,43,44,45,46):
            for method in ("ARISE","B","C"):
                rank = 1 if method == "ARISE" or method == "C" and event < 6 else 2
                result.append(dict(dataset=dataset, scenario=scenario, scenario_param="", method=method,
                    incident_id=str(event), seed=seed, rank_of_root=rank, invalid_by_design=False))
    return pd.DataFrame(result)


def test_seed_averaging_precedes_bootstrap_and_failed_is_zero():
    # Ten seed records represent just two events; each event's mean delta is +/-0.2.
    values = [[], []]
    for method, ranks in (("ARISE", [[1,1,1,2,2],[1,1,1,1,2]]), ("B",[[1,1,1,1,2],[1,1,1,2,2]])):
        for event, series in enumerate(ranks):
            for seed, rank in zip((42,43,44,45,46), series):
                values[0].append(dict(dataset="toy",scenario="S0",scenario_param="",method=method,
                    incident_id=str(event),seed=seed,rank_of_root=rank,invalid_by_design=False))
    stats = r.statistics(pd.DataFrame(values[0]), metrics=["PR@1"])
    b = stats[stats.method=="B"].iloc[0]
    assert b.events==2 and np.isclose(b.delta,0)
    np.testing.assert_allclose([b.ci_low,b.ci_high],[-20,20])
    np.testing.assert_array_equal(r.metric_values([1,2,6,np.inf,np.nan],"MAP@5"),[1,.5,0,0,0])


def test_holm_family_is_one_dataset_scenario_metric():
    solo = r.statistics(rows(),metrics=["PR@1"])
    combined = r.statistics(pd.concat([rows(),rows("other"),rows("toy","S2")]),metrics=["PR@1"])
    current=combined[(combined.dataset=="toy") & (combined.scenario=="S0")]
    np.testing.assert_array_equal(solo.p_holm.to_numpy(),current.p_holm.to_numpy())
    assert solo[solo.method=="ARISE"].p_raw.iloc[0]==1
    assert solo[solo.method=="B"].p_holm.iloc[0] < .01


def test_actual_deletion_axis_uses_cell_counts_and_auc_span():
    rates=pd.DataFrame([dict(dataset="toy",seed=42,scenario="S2",scenario_param="0.50",deleted=100,original_available=100),
        dict(dataset="toy",seed=42,scenario="S2",scenario_param="0.50",deleted=0,original_available=1)])
    assert r.realized_x(rates,"toy",42,"S0","")==0
    assert np.isclose(r.realized_x(rates,"toy",42,"S2","0.50"),100/101)
    # A flat50% curve has normalized AUC50 regardless of actual endpoint.
    scenes=[("S0",""),("S2","0.10"),("S2","0.25"),("S2","0.50"),("S2","0.75")]
    frame=[]; rate_rows=[]
    for scenario,param in scenes:
        for event,rank in enumerate((1,2)):
            frame.append(dict(dataset="toy",method="ARISE",seed=42,scenario=scenario,scenario_param=param,rank_of_root=rank))
            rate_rows.append(dict(dataset="toy",seed=42,scenario=scenario,scenario_param=param,deleted=0 if not param else float(param)*100,original_available=100))
    points, auc=r.degradation(pd.DataFrame(frame),pd.DataFrame(rate_rows),seeds=[42])
    assert np.isclose(auc[auc.metric=="PR@1"]["mean"].iloc[0],50)
    assert np.isclose(auc.x_span.iloc[0],.75)
