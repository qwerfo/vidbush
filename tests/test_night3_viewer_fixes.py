"""Causes found by the night 3 viewer checks: an AI still kept its season and year, a tiny bar stays visible."""
from vidbush.visuals.ai_images import setting_words


def test_the_ai_prompt_rewrite_must_keep_the_season_and_year():
    assert setting_words("man after a bar fight (Anchorage, Alaska, 1958, summer)") == ["summer", "1958"]
    assert setting_words("x (1950s, winter)") == ["winter", "1950s"]
    assert setting_words("crowd (Manhattan, present day)") == []
