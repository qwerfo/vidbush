"""API: health, settings and seat status, brands round trip, style profiles, UI serving."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from vidbush.api.app import create_app
from vidbush.config import REPO_ROOT


@pytest.fixture
def client(settings, tmp_path):
    """Client whose config folder is a temp copy, so brand writes never touch the repo."""
    cfg = tmp_path / "config"
    shutil.copytree(Path(REPO_ROOT) / "config", cfg)
    settings.config_folder = str(cfg)
    with TestClient(create_app(settings, start_workers=False)) as c:
        yield c


def test_health_and_settings(client):
    assert client.get("/api/health").json() == {"ok": True, "workers": False, "offline": True}
    s = client.get("/api/settings").json()
    assert s["offline"] is True
    assert set(s["mock_roles"]) == set(s["seats"]) and "tts" in s["seats"]
    assert s["seats"]["writer"]["real"] is False
    assert s["doctor"]["binaries"]["ffmpeg"] is True and s["doctor"]["fonts"]
    assert s["render"]["width"] == 1920 and "enabled" in s["sources"] and "threshold" in s["judge"]


def test_brand_round_trip(client, settings):
    brands = client.get("/api/brands").json()
    assert any(b["id"] == "default" for b in brands)
    b = client.get("/api/brands/default").json()
    b.update(name="History Vault", niche_notes="Cold war deep dives", blocklist=["badstock.net"],
             intro_path="assets/intro.mp4", language="en")
    b["voice"]["voice_id"] = "abc123"
    r = client.put("/api/brands/history_vault", json=b)
    assert r.status_code == 200, r.text
    assert r.json()["id"] == "history_vault"
    on_disk = yaml.safe_load((Path(settings.config_folder) / "brands" / "history_vault.yaml").read_text())
    assert on_disk["name"] == "History Vault" and on_disk["voice"]["voice_id"] == "abc123"
    back = client.get("/api/brands/history_vault").json()
    assert back == r.json() and back["blocklist"] == ["badstock.net"]
    assert "history_vault" in [x["id"] for x in client.get("/api/brands").json()]


@pytest.mark.parametrize("brand_id,body,code", [
    ("good", {"voice": {"stability": "very"}}, 422),
    ("good", {"style_profile_id": "does_not_exist"}, 422),
    ("bad.id", {}, 400),
])
def test_brand_validation(client, settings, brand_id, body, code):
    assert client.put(f"/api/brands/{brand_id}", json=body).status_code == code
    assert not (Path(settings.config_folder) / "brands" / "good.yaml").exists()


def test_brand_missing_is_404(client):
    assert client.get("/api/brands/nope").status_code == 404


def test_style_profiles(client):
    profiles = {p["id"]: p for p in client.get("/api/style_profiles").json()}
    assert {"documentary_default", "top10_default"} <= set(profiles)
    assert profiles["top10_default"]["top10"] is True
    assert "motion_graphic" in profiles["documentary_default"]["visual_mix"]


def test_unknown_api_path_is_json_404(client):
    r = client.get("/api/does/not/exist")
    assert r.status_code == 404 and r.json()["detail"].startswith("no such endpoint")


def _app_with_root(settings, root: Path) -> TestClient:
    settings.repo_root = str(root)
    return TestClient(create_app(settings, start_workers=False))


def test_ui_missing_dist_shows_build_help(settings, tmp_path):
    with _app_with_root(settings, tmp_path) as c:
        r = c.get("/")
        assert r.status_code == 200 and "npm run build" in r.text
        assert "npm run build" in c.get("/projects/abc").text


def test_ui_serves_dist_with_spa_fallback(settings, tmp_path):
    dist = tmp_path / "ui" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>INDEX</html>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    with _app_with_root(settings, tmp_path) as c:
        assert c.get("/").text == "<html>INDEX</html>"
        js = c.get("/assets/app.js")
        assert js.text == "console.log(1)" and "immutable" in js.headers["cache-control"]
        assert c.get("/p/some_project").text == "<html>INDEX</html>"
        assert c.get("/%2e%2e/%2e%2e/etc/passwd").text == "<html>INDEX</html>"


def test_repo_ui_dist_is_built():
    """The built UI is committed so operators without Node can run it."""
    index = Path(REPO_ROOT) / "ui" / "dist" / "index.html"
    assert index.exists() and "/assets/" in index.read_text()
