"""Stage 1 play-level EPA model (NumPyro).

    ppa ~ StudentT(nu, mu, sigma[kind])
    pass:  mu = a[pass,season] + ctx.b_pass + O[off] - D[def] + theta_pass[passer]
                + 1{target known} (k[season] + theta_recv[target])          (no target on sacks)
    rush:  mu = a[rush,season] + ctx.b_rush + O[off] - D[def] + theta_rush[rusher]
    theta_role[p,s,g] = alpha_role[position] + c_role[p] + s_role[p,s] + g_role[p,s,g]
    c ~ N(X gamma, tau_c);  s[first] = 0;  s[next] = s[prev] + N(0, tau_s * sqrt(gap));  g ~ N(0, tau_g)

Identification (from the 2019 NUTS validation): a separate first-season effect s[first] ~ N(0, tau_s0)
carries forward through the random walk exactly like c, so c and s[first] were separated only by their
priors (recv tau_c / tau_s0 R-hat 1.15). Here c is the player's level entering his first season, with
the recruiting-informed prior, and s is the change since then. Position averages alpha are coded
relative to a reference position per role (QB passing, RB rushing, WR receiving), which removes the
exact intercept / alpha / indicator ridges.
    O, D (team-season) = conference-season mean + AR(1) team deviation (FCS teams pooled, tighter)

Unknown passers/rushers get a season-level indicator instead of a player term. No defender terms:
defense enters only through D (individual defenders are valued by the box-score model).
All hierarchical effects are non-centered by default.

Scales: mean-field SVI biases the hierarchical scales (2019 validation: AutoNormal put tau_c 6-15x
below NUTS; a centered form overshot tau_c/tau_g and collapsed O's team scale). With the scales held
at NUTS posterior means, SVI matches NUTS (player-value correlation 0.99, sd ratio 0.87-0.99). So the
as-of fits hold the within-season scales at a NUTS fit on the burn-in season (d["fixed"]).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl

from cfb.player_model.data import COVARIATES, ROLES, Dataset, transfer_indicator


@dataclass
class Levels:
    """Rows grouped by depth in a per-entity season chain (level 0 = first season seen)."""
    idx: list[np.ndarray]
    parent: list[np.ndarray]
    gap: list[np.ndarray]


REFERENCE_POSITION = {"pass": "QB", "rush": "RB", "recv": "WR"}


@dataclass
class RoleArrays:
    ref_pos: int
    n_rp: int
    n_ps: int
    n_pg: int
    n_pos: int
    X: np.ndarray            # (n_rp, n_cov)
    ps_rp: np.ndarray        # player of each player-season
    levels: Levels
    play_rp: np.ndarray      # per play (0 where unknown)
    play_ps: np.ndarray
    play_pg: np.ndarray
    play_pos: np.ndarray
    play_known: np.ndarray   # float mask


@dataclass
class ModelArrays:
    y: np.ndarray
    kind: np.ndarray
    season: np.ndarray
    ctx: np.ndarray
    off_ts: np.ndarray
    def_ts: np.ndarray
    target_term: np.ndarray
    mask: np.ndarray
    n_seasons: int
    n_ts: int
    n_cg: int
    ts_cg: np.ndarray
    ts_fcs: np.ndarray
    ts_levels: Levels
    roles: dict[str, RoleArrays]
    keys: dict[str, pl.DataFrame] = field(default_factory=dict)  # compact index -> entity keys


def _levels(parent: np.ndarray, gap: np.ndarray, order: np.ndarray) -> Levels:
    """Depth of each row in its parent chain; `order` lists rows so parents come first."""
    depth = np.zeros(len(parent), dtype=np.int64)
    for i in order:
        depth[i] = 0 if parent[i] < 0 else depth[parent[i]] + 1
    out = Levels([], [], [])
    for d in range(int(depth.max()) + 1 if len(depth) else 0):
        rows = np.flatnonzero(depth == d)
        out.idx.append(rows)
        out.parent.append(parent[rows])
        out.gap.append(np.maximum(gap[rows], 1).astype(np.float32))
    return out


def _remap(used_global: np.ndarray, n_global: int) -> np.ndarray:
    m = np.full(n_global, -1, dtype=np.int64)
    m[used_global] = np.arange(len(used_global))
    return m


def make_arrays(ds: Dataset, plays: pl.DataFrame, *, compact: bool, mask: np.ndarray | None = None,
                as_of: datetime | None = None) -> ModelArrays:
    """Arrays for `plays`. compact=True keeps only entities referenced by these plays (NUTS on one
    season); compact=False keeps every entity so parameter shapes never change (as-of SVI)."""
    n = plays.height
    keys: dict[str, pl.DataFrame] = {}

    ts_all = ds.team_seasons
    ts_used = np.unique(np.concatenate([plays["off_ts"].to_numpy(), plays["def_ts"].to_numpy()])) if compact \
        else np.arange(ts_all.height)
    ts_map = _remap(ts_used, ts_all.height)
    ts = ts_all[ts_used.tolist()]
    cg_used = np.unique(ts["cg_idx"].to_numpy())
    cg_map = _remap(cg_used, ds.conf_groups.height)
    ts_parent = ts["parent"].to_numpy()
    ts_parent = np.where(ts_parent >= 0, ts_map[np.maximum(ts_parent, 0)], -1)
    ts_order = np.argsort(ts["season"].to_numpy(), kind="stable")
    keys["team_seasons"] = ts.select("team_id", "season", "conf_group", "is_fcs").with_row_index("idx")

    roles = {}
    for role in ROLES:
        rp_g, ps_g, pg_g = (plays[f"{role}_rp"], plays[f"{role}_ps"], plays[f"{role}_pg"])
        known = rp_g.is_not_null().to_numpy()
        players, pseasons, pgames = ds.players[role], ds.player_seasons[role], ds.player_games[role]
        if compact:
            rp_used = np.unique(rp_g.drop_nulls().to_numpy())
            ps_used = np.unique(ps_g.drop_nulls().to_numpy())
            pg_used = np.unique(pg_g.drop_nulls().to_numpy())
        else:
            rp_used, ps_used, pg_used = (np.arange(players.height), np.arange(pseasons.height), np.arange(pgames.height))
        rp_map, ps_map, pg_map = (_remap(rp_used, players.height), _remap(ps_used, pseasons.height),
                                  _remap(pg_used, pgames.height))
        ps = pseasons[ps_used.tolist()]
        parent = ps["parent"].to_numpy()
        parent = np.where(parent >= 0, ps_map[np.maximum(parent, 0)], -1)
        cov = players[rp_used.tolist()].select(COVARIATES).to_numpy().astype(np.float32)
        trans = transfer_indicator(ds, role, as_of)[rp_used][:, None]
        roles[role] = RoleArrays(
            ref_pos=ds.positions[role].index(REFERENCE_POSITION[role]),
            n_rp=len(rp_used), n_ps=len(ps_used), n_pg=len(pg_used), n_pos=len(ds.positions[role]),
            X=np.concatenate([cov, trans], axis=1),
            ps_rp=rp_map[ps["rp_idx"].to_numpy()],
            levels=_levels(parent, ps["gap"].to_numpy(), np.argsort(ps["season"].to_numpy(), kind="stable")),
            play_rp=np.where(known, rp_map[rp_g.fill_null(0).to_numpy()], 0),
            play_ps=np.where(known, ps_map[ps_g.fill_null(0).to_numpy()], 0),
            play_pg=np.where(known, pg_map[pg_g.fill_null(0).to_numpy()], 0),
            play_pos=plays[f"{role}_pos"].fill_null(0).to_numpy(),
            play_known=known.astype(np.float32),
        )
        keys[f"players_{role}"] = players[rp_used.tolist()].select("athlete_id").with_row_index("idx")
        keys[f"player_seasons_{role}"] = ps.select("athlete_id", "season", "position_group").with_row_index("idx")

    return ModelArrays(
        y=plays["ppa"].to_numpy().astype(np.float32),
        kind=plays["play_kind"].to_numpy().astype(np.int32),
        season=plays["season_idx"].to_numpy().astype(np.int32),
        ctx=plays.select(ds.context_columns).to_numpy().astype(np.float32),
        off_ts=ts_map[plays["off_ts"].to_numpy()],
        def_ts=ts_map[plays["def_ts"].to_numpy()],
        target_term=plays["has_target_term"].to_numpy().astype(np.float32),
        mask=np.ones(n, dtype=bool) if mask is None else mask,
        n_seasons=len(ds.seasons), n_ts=len(ts_used), n_cg=len(cg_used),
        ts_cg=cg_map[ts["cg_idx"].to_numpy()], ts_fcs=ts["is_fcs"].to_numpy().astype(np.float32),
        ts_levels=_levels(ts_parent, ts["gap"].to_numpy(), ts_order),
        roles=roles, keys=keys,
    )


# ---------- static sizes and device data ----------
# The model takes (static, data): static holds Python ints (shapes), data holds every array as a
# JAX array. Passing data as an argument to jitted functions keeps it out of the compiled program
# (closing over host arrays embeds them as constants, which exhausted memory on the full dataset).

@dataclass(frozen=True)
class Static:
    n_seasons: int
    n_ts: int
    n_cg: int
    n_ctx: int
    n_plays: int
    team_levels: int
    roles: tuple  # ((role, n_rp, n_ps, n_pg, n_pos, n_cov, n_levels, ref_pos), ...)
    # Centered player effects (c, season increments, g sampled directly) for SVI: mean-field VI on the
    # non-centered form collapsed tau_c 6-15x below NUTS on the 2019 validation season.
    centered: bool = False

    def role(self, name: str) -> dict:
        for r in self.roles:
            if r[0] == name:
                return dict(zip(("role", "n_rp", "n_ps", "n_pg", "n_pos", "n_cov", "n_levels", "ref_pos"), r))
        raise KeyError(name)


def _levels_data(lv: Levels) -> dict:
    return {"idx": [jnp.asarray(x) for x in lv.idx], "parent": [jnp.asarray(x) for x in lv.parent],
            "gap": [jnp.asarray(x) for x in lv.gap]}


def to_device(a: ModelArrays, centered: bool = False) -> tuple[Static, dict]:
    static = Static(
        centered=centered,
        n_seasons=a.n_seasons, n_ts=a.n_ts, n_cg=a.n_cg, n_ctx=a.ctx.shape[1], n_plays=len(a.y),
        team_levels=len(a.ts_levels.idx),
        roles=tuple((role, r.n_rp, r.n_ps, r.n_pg, r.n_pos, r.X.shape[1], len(r.levels.idx), r.ref_pos)
                    for role, r in a.roles.items()),
    )
    data = {
        "y": jnp.asarray(a.y), "kind": jnp.asarray(a.kind), "season": jnp.asarray(a.season),
        "ctx": jnp.asarray(a.ctx), "off_ts": jnp.asarray(a.off_ts), "def_ts": jnp.asarray(a.def_ts),
        "target_term": jnp.asarray(a.target_term), "mask": jnp.asarray(a.mask),
        "ts_cg": jnp.asarray(a.ts_cg), "ts_fcs": jnp.asarray(a.ts_fcs), "ts_levels": _levels_data(a.ts_levels),
        "roles": {role: {
            "X": jnp.asarray(r.X), "levels": _levels_data(r.levels), "play_rp": jnp.asarray(r.play_rp),
            "play_ps": jnp.asarray(r.play_ps), "play_pg": jnp.asarray(r.play_pg), "play_pos": jnp.asarray(r.play_pos),
            "play_known": jnp.asarray(r.play_known)} for role, r in a.roles.items()},
    }
    return static, data


# ---------- model ----------

def _chain(z: jnp.ndarray, levels: dict, first_scale, step_scale, decay=None) -> jnp.ndarray:
    """Random walk (decay=None) or AR(1) along each entity's season chain, non-centered."""
    out = jnp.zeros_like(z)
    for d, (idx, parent, gap) in enumerate(zip(levels["idx"], levels["parent"], levels["gap"])):
        if d == 0:
            out = out.at[idx].set(_take(first_scale, idx) * z[idx])
        else:
            carry = out[parent] if decay is None else (decay ** gap) * out[parent]
            out = out.at[idx].set(carry + _take(step_scale, idx) * jnp.sqrt(gap) * z[idx])
    return out


def _take(scale, idx):
    return scale[idx] if getattr(scale, "ndim", 0) else scale


# Hierarchical scales that go through _hyper. Within-season scales are identified by one season of
# plays; the season-to-season ones (random-walk step, team AR(1)) need at least two.
SEASON_STEP_SCALES = ("O_tau_team", "O_rho", "D_tau_team", "D_rho", *(f"{r}_tau_s" for r in ROLES))
WITHIN_SEASON_SCALES = (*(f"{u}_{p}" for u in ("O", "D") for p in ("tau_conf", "tau_team0", "fcs_scale")),
                        *(f"{r}_{p}" for r in ROLES for p in ("tau_c", "tau_g")))


def _hyper(name: str, prior, d: dict):
    """A hierarchical scale: sampled, or held at d["fixed"][name] (empirical Bayes; mean-field SVI
    biases these scales, see fit.py)."""
    fixed = d.get("fixed", {})
    if name in fixed:
        return numpyro.deterministic(name, fixed[name])
    return numpyro.sample(name, prior)


def _units(name: str, st: Static, d: dict) -> jnp.ndarray:
    tau_conf = _hyper(f"{name}_tau_conf", dist.HalfNormal(0.15), d)
    conf = tau_conf * numpyro.sample(f"{name}_conf_z", dist.Normal(0, 1).expand([st.n_cg]).to_event(1))
    tau0 = _hyper(f"{name}_tau_team0", dist.HalfNormal(0.15), d)
    tau = _hyper(f"{name}_tau_team", dist.HalfNormal(0.1), d)
    fcs_scale = _hyper(f"{name}_fcs_scale", dist.Beta(2.0, 5.0), d)  # FCS spread as a fraction of FBS
    rho = _hyper(f"{name}_rho", dist.Beta(3.0, 2.0), d)
    shrink = 1.0 - d["ts_fcs"] * (1.0 - fcs_scale)
    z = numpyro.sample(f"{name}_z", dist.Normal(0, 1).expand([st.n_ts]).to_event(1))
    dev = _chain(z, d["ts_levels"], tau0 * shrink, tau * shrink, decay=rho)
    return numpyro.deterministic(name, conf[d["ts_cg"]] + dev)


def _player_effects(role: str, st: Static, d: dict, with_games: bool = True):
    """alpha (position averages), c (career), s (season random walk), g (game) for one role."""
    n, r = st.role(role), d["roles"][role]
    free = numpyro.sample(f"{role}_alpha_free", dist.Normal(0, 0.3).expand([n["n_pos"] - 1]).to_event(1))
    ref = n["ref_pos"]
    alpha = numpyro.deterministic(f"{role}_alpha", jnp.concatenate([free[:ref], jnp.zeros(1), free[ref:]]))
    gamma = numpyro.sample(f"{role}_gamma", dist.Normal(0, 0.1).expand([n["n_cov"]]).to_event(1))
    tau_c = _hyper(f"{role}_tau_c", dist.HalfNormal(0.1), d)
    tau_s = _hyper(f"{role}_tau_s", dist.HalfNormal(0.05), d)
    tau_g = _hyper(f"{role}_tau_g", dist.HalfNormal(0.05), d)
    if st.centered:
        c = numpyro.sample(f"{role}_c", dist.Normal(r["X"] @ gamma, tau_c).to_event(1))
        # First-season slots are unused (s[first] = 0); a unit prior keeps them out of tau_s.
        first = jnp.zeros(n["n_ps"], dtype=bool).at[r["levels"]["idx"][0]].set(True)
        eps = numpyro.sample(f"{role}_s_eps", dist.Normal(0, jnp.where(first, 1.0, tau_s)).to_event(1))
        s = numpyro.deterministic(f"{role}_s", _chain(eps, r["levels"], 0.0, 1.0))
        if not with_games:
            return alpha, c, s, None
        return alpha, c, s, numpyro.sample(f"{role}_g", dist.Normal(0, tau_g).expand([n["n_pg"]]).to_event(1))
    c = r["X"] @ gamma + tau_c * numpyro.sample(f"{role}_c_z", dist.Normal(0, 1).expand([n["n_rp"]]).to_event(1))
    s = _chain(numpyro.sample(f"{role}_s_z", dist.Normal(0, 1).expand([n["n_ps"]]).to_event(1)), r["levels"], 0.0, tau_s)
    numpyro.deterministic(f"{role}_c", c)
    numpyro.deterministic(f"{role}_s", s)
    if not with_games:
        return alpha, c, s, None
    g = tau_g * numpyro.sample(f"{role}_g_z", dist.Normal(0, 1).expand([n["n_pg"]]).to_event(1))
    return alpha, c, s, g


def _player_terms(role: str, st: Static, d: dict) -> jnp.ndarray:
    alpha, c, s, g = _player_effects(role, st, d)
    r = d["roles"][role]
    return alpha[r["play_pos"]] + c[r["play_rp"]] + s[r["play_ps"]] + g[r["play_pg"]]


def effects_model(st: Static, d: dict) -> None:
    """The latent effects only (units, alpha, c, s): sample values from a fitted guide without plays."""
    _units("O", st, d)
    _units("D", st, d)
    for role in ROLES:
        _player_effects(role, st, d, with_games=False)


def model(st: Static, d: dict, observe: bool = True) -> None:
    S = st.n_seasons
    intercept = numpyro.sample("intercept", dist.Normal(0, 0.5).expand([2, S]).to_event(2))
    beta = numpyro.sample("beta", dist.Normal(0, 0.5).expand([2, st.n_ctx]).to_event(2))
    kappa = numpyro.sample("kappa_target_known", dist.Normal(0, 0.5).expand([S]).to_event(1))
    iota_pass = numpyro.sample("iota_passer_unknown", dist.Normal(0, 0.5).expand([S]).to_event(1))
    iota_rush = numpyro.sample("iota_rusher_unknown", dist.Normal(0, 0.5).expand([S]).to_event(1))
    O = _units("O", st, d)
    D = _units("D", st, d)
    theta = {role: _player_terms(role, st, d) for role in ROLES}

    kind, season, ctx = d["kind"], d["season"], d["ctx"]
    is_pass = kind == 0
    pk, rk, tk = (d["roles"][r]["play_known"] for r in ("pass", "rush", "recv"))
    player = jnp.where(is_pass,
                       pk * theta["pass"] + (1 - pk) * iota_pass[season],
                       rk * theta["rush"] + (1 - rk) * iota_rush[season])
    target = d["target_term"] * tk * (kappa[season] + theta["recv"])
    mu = (intercept[kind, season] + jnp.where(is_pass, ctx @ beta[0], ctx @ beta[1])
          + O[d["off_ts"]] - D[d["def_ts"]] + player + target)
    if not observe:
        numpyro.deterministic("mu", mu)
    _likelihood(st, d, mu, observe)


def _likelihood(st: Static, d: dict, mu: jnp.ndarray, observe: bool) -> None:
    nu = 1.0 + numpyro.sample("nu_minus_1", dist.Gamma(2.0, 0.2))
    sigma = numpyro.sample("sigma", dist.HalfNormal(2.0).expand([2]).to_event(1))
    with numpyro.plate("plays", st.n_plays), numpyro.handlers.mask(mask=d["mask"]):
        numpyro.sample("y", dist.StudentT(nu, mu, sigma[d["kind"]]), obs=d["y"] if observe else None)


def context_model(st: Static, d: dict, observe: bool = True) -> None:
    """Baseline: intercepts and context only (no units, no players)."""
    intercept = numpyro.sample("intercept", dist.Normal(0, 0.5).expand([2, st.n_seasons]).to_event(2))
    beta = numpyro.sample("beta", dist.Normal(0, 0.5).expand([2, st.n_ctx]).to_event(2))
    kind = d["kind"]
    mu = intercept[kind, d["season"]] + jnp.where(kind == 0, d["ctx"] @ beta[0], d["ctx"] @ beta[1])
    _likelihood(st, d, mu, observe)
