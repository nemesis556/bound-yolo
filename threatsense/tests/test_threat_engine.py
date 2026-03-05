from app.threat_engine import ThreatScoringEngine


def test_scoring_levels():
    engine = ThreatScoringEngine()
    low = engine.score(["crowd_alert"])
    med = engine.score(["running", "loitering"])
    critical = engine.score(["abandoned_object", "restricted_zone_entry"])

    assert low.level == "LOW"
    assert med.level == "MEDIUM"
    assert critical.level == "CRITICAL"
