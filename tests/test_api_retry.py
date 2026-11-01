"""A gateway error from the data API is waited out, and a cached copy beats failing the build."""

from __future__ import annotations

import json

import pytest

from mri.ingest import cfbd


class _Response:
    def __init__(self, status: int, body=None) -> None:
        self.status_code, self.ok, self.text, self.headers = status, status < 400, "bad gateway", {}
        self._body = body

    def json(self):
        return self._body


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(cfbd, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cfbd, "_FETCHED", set())
    monkeypatch.setattr(cfbd.time, "sleep", lambda s: None)
    monkeypatch.setenv("CFBD_API_KEY", "test")
    return tmp_path


def test_a_502_is_retried_until_the_api_recovers(api, monkeypatch) -> None:
    answers = iter([_Response(502), _Response(502), _Response(200, [{"ok": 1}])])
    monkeypatch.setattr(cfbd.requests, "get", lambda *a, **k: next(answers))
    assert cfbd.request("/games", year=2026) == [{"ok": 1}]


def test_a_lasting_502_serves_the_cached_copy(api, monkeypatch) -> None:
    path = cfbd._cache_path("/games", {"year": 2026})
    path.write_text(json.dumps([{"cached": 1}]))
    monkeypatch.setattr(cfbd.requests, "get", lambda *a, **k: _Response(502))
    assert cfbd.request("/games", refresh=True, year=2026) == [{"cached": 1}]


def test_a_lasting_502_with_nothing_cached_still_fails(api, monkeypatch) -> None:
    monkeypatch.setattr(cfbd.requests, "get", lambda *a, **k: _Response(502))
    with pytest.raises(cfbd.CfbdError):
        cfbd.request("/games", year=2026)
