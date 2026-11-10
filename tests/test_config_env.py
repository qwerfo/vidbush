"""Env files: .env then .env.local (.env.local wins), process env wins over both, tests read neither."""
from __future__ import annotations

import shutil
from pathlib import Path

from vidbush.config import REPO_ROOT, load_settings


def _root(tmp_path: Path) -> Path:
    shutil.copytree(REPO_ROOT / "config", tmp_path / "config")
    (tmp_path / ".env").write_text("VB_TEST_A=1\nVB_TEST_B=1\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("VB_TEST_B=2\n", encoding="utf-8")
    return tmp_path


def test_env_local_wins_over_env_and_process_env_wins_over_both(tmp_path: Path, monkeypatch) -> None:
    root = _root(tmp_path)
    monkeypatch.delenv("VB_TEST_A", raising=False)
    monkeypatch.delenv("VB_TEST_B", raising=False)
    s = load_settings(root)
    assert s.key("VB_TEST_A") == "1" and s.key("VB_TEST_B") == "2"
    monkeypatch.setenv("VB_TEST_B", "3")
    assert load_settings(root).key("VB_TEST_B") == "3"


def test_use_dotenv_false_reads_neither_file(tmp_path: Path, monkeypatch) -> None:
    root = _root(tmp_path)
    monkeypatch.delenv("VB_TEST_A", raising=False)
    monkeypatch.delenv("VB_TEST_B", raising=False)
    s = load_settings(root, use_dotenv=False)
    assert s.key("VB_TEST_A") == "" and s.key("VB_TEST_B") == ""
