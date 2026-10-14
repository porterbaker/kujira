"""Compare live responses with the vendored cfbd models (field names, presence, nullability, types).

The models in ``vendor/cfbd-python/cfbd/models`` are parsed as source (AST), never imported.
In these generated models every field is declared with ``default=...`` (it must be present).
Nullability comes only from ``Optional[...]``.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from cfb.config import ROOT

VENDOR_MODELS = ROOT / "vendor" / "cfbd-python" / "cfbd" / "models"

MODEL_BY_SLUG = {
    "info": "user_info",
    "plays_types": "play_type",
    "plays_stats_types": "play_stat_type",
    "calendar": "calendar_week",
    "games": "game",
    "plays": "play",
    "passing_plays": "passing_play",
    "rushing_plays": "rushing_play",
    "plays_stats": "play_stat",
    "games_players": "game_player_stats",
    "ppa_players_games": "player_game_predicted_points_added",
    "lines": "betting_game",
    "metrics_wp_pregame": "pregame_win_probability",
    "roster": "roster_player",
    "recruiting_players": "recruit",
    "player_portal": "player_transfer",
    "draft_picks": "draft_pick",
    "venues": "venue",
    "coaches": "coach",
    "teams": "team",
    "talent": "team_talent",
    "player_returning": "returning_production",
    "ppa_players_season": "player_season_predicted_points_added",
    "wepa_players_passing": "player_weighted_epa",
    "wepa_players_rushing": "player_weighted_epa",
    "wepa_players_kicking": "kicker_paar",
    "ratings_sp": "team_sp",
    "ratings_srs": "team_srs",
    "ratings_elo": "team_elo",
    "ratings_fpi": "team_fpi",
    "games_weather": "game_weather",
}


@dataclass(frozen=True)
class FieldSpec:
    name: str  # wire name (alias)
    annotation: str
    required: bool
    nullable: bool


@dataclass
class SchemaDiff:
    model: str
    rows: int
    missing: list[str] = field(default_factory=list)  # in model, absent from some rows
    extra: list[str] = field(default_factory=list)  # in data, not in model
    null_violations: dict[str, int] = field(default_factory=dict)  # non-Optional fields with nulls
    type_mismatches: dict[str, str] = field(default_factory=dict)  # field -> "expected X, saw Y"

    @property
    def clean(self) -> bool:
        return not (self.missing or self.extra or self.null_violations or self.type_mismatches)


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


@cache
def model_fields(module: str, models_dir: Path = VENDOR_MODELS) -> dict[str, FieldSpec]:
    tree = ast.parse((models_dir / f"{module}.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    out = {}
    for stmt in cls.body:
        if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
            continue
        alias, required = stmt.target.id, False
        if isinstance(stmt.value, ast.Call):
            if stmt.value.args and isinstance(stmt.value.args[0], ast.Constant) and stmt.value.args[0].value is ...:
                required = True
            for kw in stmt.value.keywords:
                if kw.arg == "alias":
                    alias = kw.value.value
                if kw.arg == "default" and isinstance(kw.value, ast.Constant) and kw.value.value is ...:
                    required = True
        ann = ast.unparse(stmt.annotation)
        out[alias] = FieldSpec(alias, ann, required, ann.startswith("Optional["))
    return out


@cache
def _is_enum(type_name: str, models_dir: Path = VENDOR_MODELS) -> bool:
    path = models_dir / f"{_snake(type_name)}.py"
    return path.is_file() and "(str, Enum)" in path.read_text()


def _expected(annotation: str) -> str:
    base = annotation[len("Optional["):-1] if annotation.startswith("Optional[") else annotation
    if base in ("StrictStr", "datetime", "date"):
        return "str"
    if base == "StrictInt":
        return "int"
    if base == "StrictBool":
        return "bool"
    if base in ("StrictFloat", "Union[StrictFloat, StrictInt]"):
        return "number"
    if base.startswith(("conlist", "List[")):
        return "list"
    if re.fullmatch(r"[A-Z]\w*", base):
        return "str" if _is_enum(base) else "dict"
    return "any"


def _actual(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return {str: "str", list: "list", dict: "dict"}.get(type(value), type(value).__name__)


def _type_ok(expected: str, actual: str) -> bool:
    return expected == "any" or expected == actual or (expected == "number" and actual in ("int", "float"))


def compare(rows: list[dict] | dict, module: str) -> SchemaDiff:
    rows = rows if isinstance(rows, list) else [rows]
    fields = model_fields(module)
    diff = SchemaDiff(module, len(rows))
    seen: set[str] = set()
    missing: dict[str, int] = {}
    nulls: dict[str, int] = {}
    mismatches: dict[str, set[str]] = {}
    for row in rows:
        seen.update(row)
        for name, spec in fields.items():
            if name not in row:
                if spec.required:
                    missing[name] = missing.get(name, 0) + 1
                continue
            value = row[name]
            if value is None:
                if not spec.nullable:
                    nulls[name] = nulls.get(name, 0) + 1
                continue
            expected = _expected(spec.annotation)
            if not _type_ok(expected, _actual(value)):
                mismatches.setdefault(name, set()).add(_actual(value))
    diff.missing = sorted(missing)
    diff.extra = sorted(seen - fields.keys())
    diff.null_violations = dict(sorted(nulls.items()))
    diff.type_mismatches = {
        name: f"expected {_expected(fields[name].annotation)}, saw {'/'.join(sorted(types))}"
        for name, types in sorted(mismatches.items())
    }
    return diff
