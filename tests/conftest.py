import gzip
import json
import socket
from pathlib import Path

import pytest

from cfb.config import Paths, Settings
from cfb.ingest import raw_store


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """The suite runs offline: any socket connection fails loudly."""

    def blocked(*args, **kwargs):
        raise RuntimeError("network access is blocked in tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings rooted in a temp dir, so tests never see the real data/ directory."""
    return Settings(paths=Paths(root=tmp_path))


FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str):
    """Trimmed live responses saved by `cfb ingest probe --report` (see reports/phase1a_probe.md)."""
    return json.loads((FIXTURES / f"{name}.json").read_text())


def write_raw(settings: Settings, slug: str, partition: raw_store.Partition, payload) -> Path:
    path = raw_store.response_path(settings.paths.raw, slug, partition)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path
