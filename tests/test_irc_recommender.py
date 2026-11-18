from rodic_inference.irc_recommender import (
    DISCLAIMER,
    normalise_class,
    recommend,
    severity_counts,
    severity_from_crack_width,
    worst_severity,
)


def test_crack_width_bands():
    assert severity_from_crack_width(0.1) == "cosmetic"
    assert severity_from_crack_width(0.2) == "watch"   # boundary belongs to the upper band
    assert severity_from_crack_width(0.49) == "watch"
    assert severity_from_crack_width(1.0) == "significant"
    assert severity_from_crack_width(2.0) == "urgent"
    assert severity_from_crack_width(5.0) == "urgent"


def test_hairline_vs_two_mm_crack():
    assert recommend("Crack", 0.9, 0.2).severity == "watch"
    two_mm = recommend("Crack", 0.9, 2.0)
    assert two_mm.severity == "urgent"
    assert any("structural audit" in a.lower() for a in two_mm.actions)


def test_significant_crack_mentions_epoxy():
    rec = recommend("crack", 0.8, 1.0)
    assert any("epoxy" in a.lower() for a in rec.actions)


def test_unknown_width_is_provisional_not_guessed():
    rec = recommend("Crack", 0.9, None)
    assert rec.severity == "watch"
    assert "provisional" in rec.severity_basis
    assert any("width" in n.lower() for n in rec.notes)


def test_unresolvable_width_is_flagged_as_upper_bound():
    rec = recommend("Crack", 0.9, 1.5, width_resolvable=False)
    assert "provisional" in rec.severity_basis
    assert any("upper bound" in n for n in rec.notes)


def test_class_aliases():
    assert normalise_class("ExposedBars") == "exposed_bars"
    assert normalise_class("Corrosion Stain") == "corrosion_stain"
    assert normalise_class("Background") == "none"
    assert normalise_class(None) == "unclassified"
    assert normalise_class("something odd") == "unclassified"


def test_background_has_no_action_and_unclassified_needs_human():
    assert recommend("Background").actions == []
    rec = recommend(None)
    assert rec.severity == "watch" and rec.actions


def test_every_recommendation_requires_engineer_review():
    for cls in ["Crack", "Spallation", "Efflorescence", "ExposedBars", "CorrosionStain", "Background", None]:
        rec = recommend(cls, 0.9, 0.3 if cls == "Crack" else None)
        assert rec.needs_engineer_review is True
        assert rec.disclaimer == DISCLAIMER


def test_low_confidence_note():
    rec = recommend("Spallation", 0.2)
    assert any("below" in n for n in rec.notes)


def test_worst_severity_and_counts():
    recs = [recommend("Efflorescence"), recommend("ExposedBars"), recommend("Crack", None, 0.3)]
    assert worst_severity(recs) == "urgent"
    assert severity_counts(recs)["urgent"] == 1
    assert sum(severity_counts(recs).values()) == 3
