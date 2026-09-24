"""End-to-end TM process on synthetic data: invariants a validator would check."""
from __future__ import annotations

import pytest

from src.tm import explain, kpis, pipeline, synthetic, thresholds
from src.tm.ars import ARS_FEATURES

FAST_EBM = {"interactions": 2, "outer_bags": 2, "n_jobs": 1}


@pytest.fixture(scope="module")
def art():
    return pipeline.run(synthetic.generate(n_customers=1200, seed=3), ebm_params=FAST_EBM)


def test_segments_split_by_entity_type(art):
    segs = set(art.segment_profile.segment)
    assert any(s.startswith("IND_") for s in segs) and any(s.startswith("CORP_") for s in segs)


def test_tuning_respects_recall_floor(art):
    tuned = art.tuned_thresholds[art.tuned_thresholds.status == "TUNED"]
    assert (tuned.relative_recall >= 0.95).all()


def test_policy_floor_never_breached(art):
    floors = {s.scenario_id: s.min_floor for s in thresholds.SCENARIO_REGISTRY}
    t = art.tuned_thresholds
    assert (t.threshold >= t.scenario_id.map(floors)).all()


def test_hard_overrides_never_auto_close(art):
    s = art.scored_alerts
    assert not ((s.route == "AUTO_CLOSE") & s.hard_override).any()


def test_sar_leakage_within_tolerance_on_validation(art):
    assert art.ars_model.metrics["sar_leakage"] <= art.ars_model.miss_tolerance + 1e-9


def test_btl_sample_never_exceeds_band(art):
    assert (art.btl_plan.sample_size <= art.btl_plan.band_population).all()


def test_local_explanation_reconstructs_score(art):
    """EBM is additive: intercept + contributions = logit(score)."""
    import numpy as np
    s = art.scored_alerts
    X = s[ARS_FEATURES]
    loc = explain.local_explanation(art.ars_model.ebm, X, 0)
    logit = loc.attrs["intercept"] + loc.contribution.sum()
    assert abs(1 / (1 + np.exp(-logit)) - s.ars_score.iloc[0]) < 1e-4


def test_explanation_text_mentions_cutoff(art):
    s = art.scored_alerts
    txt = explain.explanation_text(art.ars_model.ebm, s[ARS_FEATURES], 0,
                                   s.ars_score.iloc[0], art.ars_model.cutoff)
    assert "cutoff" in txt and "log-odds" in txt


def test_psi_zero_on_identical_distribution(art):
    x = art.scored_alerts.ars_score
    assert kpis.psi(x, x) < 1e-6
