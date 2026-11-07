import numpy as np
import pandas as pd
import pytest

from exp_revision import t8_reports


def cohort():
    rows=[]
    for method,seeds in (("ARISE",range(42,47)),("ReAct",range(42,45))):
        for seed in seeds:
            for event in range(2):
                hit=(seed%2==0) if method=="ARISE" and event==0 else (seed%2==1) if method=="ARISE" else event==0
                rows.append(dict(dataset="test",scenario="S0",scenario_param="",method=method,seed=seed,
                    incident_id=str(event),root_rank=1. if hit else np.inf))
    return pd.DataFrame(rows)


def test_mixed_repeats_pair_events_after_averaging():
    stats=t8_reports.statistics(cohort())
    result=stats[(stats.method=="ReAct")&(stats.metric=="PR@1")].iloc[0]
    assert result.repeats==3 and result.events==2
    assert result["mean"]==50. and result.sd==0.
    assert result.delta_vs_ARISE==pytest.approx(0.)
    assert result.ci_low==pytest.approx(-40.) and result.ci_high==pytest.approx(40.)
    assert result.p_raw==1. and result.p_holm==1.
    reference=stats[(stats.method=="ARISE")&(stats.metric=="PR@1")].iloc[0]
    assert reference.repeats==5 and reference.events==2


def test_missing_repetition_is_rejected():
    frame=cohort()
    with pytest.raises(ValueError,match="incomplete"):
        t8_reports.statistics(frame[~((frame.method=="ReAct")&(frame.seed==44)&(frame.incident_id=="1"))])


def test_duplicate_event_seed_is_rejected():
    frame=cohort()
    with pytest.raises(ValueError,match="duplicate"):
        t8_reports.statistics(pd.concat([frame,frame.iloc[:1]]))
