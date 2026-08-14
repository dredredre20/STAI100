"""
Unit evals — deterministic, no LLM calls, no network.
Run with: pytest evaluation/unit/ -v
"""
import pytest
from ds_integration.job_fit_prediction import get_tier, get_match_label, TIER_LABELS


class TestTierBoundaries:
    """Regression tests for the exact threshold boundaries. These are the
    most bug-prone lines in the whole pipeline — off-by-one or >= vs >
    mistakes here silently mislabel every downstream tool."""

    def test_strong_match_at_exact_threshold(self):
        assert get_match_label(0.75) == "Strong Match"

    def test_just_below_strong_threshold_is_possible(self):
        assert get_match_label(0.7499) == "Possible Match"

    def test_possible_match_at_exact_threshold(self):
        assert get_match_label(0.55) == "Possible Match"

    def test_just_below_possible_threshold_is_weak(self):
        assert get_match_label(0.5499) == "Weak Match"

    def test_perfect_score(self):
        assert get_match_label(1.0) == "Strong Match"

    def test_zero_score(self):
        assert get_match_label(0.0) == "Weak Match"

    def test_negative_score(self):
        # cosine similarity can technically be negative
        assert get_match_label(-0.2) == "Weak Match"


class TestTierLabelMapping:
    """get_tier() must always return one of the three user-facing names,
    and TIER_LABELS must stay in sync with skill_gap.py's TIER_PROMPTS keys."""

    @pytest.mark.parametrize("score,expected_tier", [
        (0.9, "Ready to Apply"),
        (0.75, "Ready to Apply"),
        (0.6, "Near Match"),
        (0.55, "Near Match"),
        (0.3, "Long-Term Upskilling"),
        (0.0, "Long-Term Upskilling"),
    ])
    def test_get_tier(self, score, expected_tier):
        assert get_tier(score) == expected_tier

    def test_all_tier_labels_are_valid_prompt_keys(self):
        """Guards against skill_gap.py's TIER_PROMPTS drifting out of sync
        with job_fit_prediction.py's TIER_LABELS — this would have caught
        the earlier bug where tier names didn't match between files."""
        from ds_integration.skill_gap import TIER_PROMPTS
        for tier_name in TIER_LABELS.values():
            assert tier_name in TIER_PROMPTS, (
                f"'{tier_name}' from TIER_LABELS has no matching prompt in TIER_PROMPTS"
            )
