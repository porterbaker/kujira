from pathlib import Path

import pytest

from cfb.api.budget import BudgetCheck
from cfb.api.client import CFBDClient
from cfb.config import API_KEY_ENV_VAR, ROOT


def test_gitignore_covers_secrets_and_data():
    lines = {line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()}
    assert ".env" in lines
    assert "data/" in lines


def test_client_repr_redacts_key():
    client = CFBDClient("super-secret-key")
    assert "super-secret-key" not in repr(client)
    assert "redacted" in repr(client)


def test_real_key_not_in_source_tree():
    env = ROOT / ".env"
    if not env.is_file():
        pytest.skip("no .env")
    key = ""
    for line in env.read_text().splitlines():
        if line.startswith(f"{API_KEY_ENV_VAR}="):
            key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        pytest.skip("no key in .env")
    files = [p for d in ("cfb", "tests") for p in (ROOT / d).rglob("*") if p.is_file() and p.suffix in {".py", ".md", ".json", ".toml"}]
    files += [ROOT / name for name in ("README.md", "PLAN.md", "pyproject.toml", ".env.example") if (ROOT / name).is_file()]
    offenders = [str(p.relative_to(ROOT)) for p in files if key in p.read_text(errors="ignore")]
    assert offenders == []


def test_network_is_blocked():
    import httpx

    with pytest.raises(RuntimeError, match="network access is blocked"):
        httpx.get("https://api.collegefootballdata.com/info", timeout=1)


def test_budget_check():
    assert BudgetCheck(planned=900, remaining=1000, buffer=100).ok
    assert not BudgetCheck(planned=901, remaining=1000, buffer=100).ok
    assert BudgetCheck(planned=10, remaining=None, buffer=100).ok
