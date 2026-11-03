"""Licence normalisation, NC/ND flags and the domain blocklist."""
from __future__ import annotations

import pytest

from vidbush.source.blocklist import Blocklist, host_of, normalise_domain
from vidbush.source.licence import is_nc_nd, is_public_domain, licence_flags, normalise_licence


@pytest.mark.parametrize("raw,expected", [
    ("CC BY-SA 4.0", "CC BY-SA 4.0"),
    ("cc-by-sa-3.0", "CC BY-SA 3.0"),
    ("CC BY 2.0", "CC BY 2.0"),
    ("http://creativecommons.org/licenses/by-nc/2.0/", "CC BY-NC 2.0"),
    ("https://creativecommons.org/licenses/by-nc-nd/4.0/", "CC BY-NC-ND 4.0"),
    ("http://creativecommons.org/publicdomain/zero/1.0/", "CC0 1.0"),
    ("CC0", "CC0 1.0"),
    ("http://creativecommons.org/publicdomain/mark/1.0/", "Public domain"),
    ("Public domain", "Public domain"),
    ("PD-US", "Public domain"),
    ("Public domain (NASA)", "Public domain (NASA)"),
    ("Creative Commons Attribution-NonCommercial-ShareAlike 4.0", "CC BY-NC-SA 4.0"),
    ("Creative Commons Attribution", "Creative Commons Attribution"),
    ("Creative Commons Attribution 3.0", "CC BY 3.0"),
    ("", "unknown"),
    ("Pexels License", "Pexels License"),
    ("CC0 (Vidbush generated fixture)", "CC0 (Vidbush generated fixture)"),
])
def test_normalise_licence(raw: str, expected: str) -> None:
    assert normalise_licence(raw) == expected


@pytest.mark.parametrize("lic,flag", [
    ("CC BY-NC 2.0", True), ("CC BY-ND 4.0", True), ("CC BY-NC-SA 4.0", True), ("CC BY-SA 4.0", False),
    ("CC0 1.0", False), ("Public domain", False), ("Pexels License", False), ("Standard YouTube License", False),
    ("Creative Commons Attribution-NoDerivs", True),
])
def test_nc_nd_detection(lic: str, flag: bool) -> None:
    assert is_nc_nd(lic) is flag


def test_licence_flags_follow_setting(settings) -> None:
    assert licence_flags("CC BY-NC 2.0", settings) == ["nc_nd"]
    settings.sources.flag_nc_nd = False
    assert licence_flags("CC BY-NC 2.0", settings) == []


def test_public_domain_helper() -> None:
    assert is_public_domain("Public domain (NASA)") and is_public_domain("CC0 1.0")
    assert not is_public_domain("CC BY 4.0")


def test_normalise_domain() -> None:
    assert normalise_domain("https://www.Example.com/path?x=1") == "example.com"
    assert normalise_domain("badstock.net") == "badstock.net"
    assert normalise_domain("  ") == ""
    assert host_of("https://cdn.Example.com/a.jpg") == "cdn.example.com"


def test_blocklist_matches_subdomains_only() -> None:
    bl = Blocklist.from_settings(["example.com"], ["www.badstock.net"])
    assert bl.blocked("https://example.com/a")
    assert bl.blocked("", "https://cdn.media.example.com/x.mp4")
    assert bl.blocked("https://badstock.net/item/1")
    assert not bl.blocked("https://notexample.com/a", "https://example.com.evil.org/")
    assert not bl.blocked("/local/file.mp4", "")
    assert not Blocklist([])
