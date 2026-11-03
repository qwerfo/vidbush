"""Recap grid: explicit payload numbers and the highlight on number 1."""
from __future__ import annotations

from vidbush.mg import builders as b
from vidbush.render import mg_python
from vidbush.render.mg_python import build_comp, recap_numbers


def test_payload_numbers_win() -> None:
    assert recap_numbers(["a", "b", "c"], False, [10, 9, 8]) == [(10, "a"), (9, "b"), (8, "c")]
    assert recap_numbers(["#3 Gatun", "Colon"], False, [3, 1]) == [(3, "Gatun"), (1, "Colon")]
    assert recap_numbers(["a", "b"], True, [5]) == [(2, "a"), (1, "b")], "ignored when lengths differ"


def test_number_one_is_highlighted(monkeypatch) -> None:
    seen: list[tuple[int, bool]] = []
    real = mg_python.RecapGrid._card

    def spy(self, w, h, num, label, top):  # noqa: ANN001
        seen.append((num, top))
        return real(self, w, h, num, label, top)

    monkeypatch.setattr(mg_python.RecapGrid, "_card", spy)
    spec = b.recap_grid_spec(["Gatun", "Colon", "Culebra"], title="Recap", duration=2.0)
    spec = spec.model_copy(update={"width": 640, "height": 360,
                                   "payload": {**spec.payload, "numbers": [3, 1, 2]}})
    build_comp(spec, False).render(10)
    assert seen == [(3, False), (1, True), (2, False)]
