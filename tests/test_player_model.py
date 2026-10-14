"""Stage 1 model: dataset construction, as-of masking, model mechanics, and a short end-to-end SVI fit
on the real-data fixtures (three games: 2014, 2024, 2025)."""

from datetime import datetime, timezone

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from cfb.clean.build import build_all
from cfb.clean.fixtures import install_fixtures
from cfb.config import Paths, Settings
from cfb.player_model import model as M
from cfb.player_model.data import ROLES, build_dataset, transfer_indicator
from cfb.player_model.fit import effects_from_samples, heldout_loglik, run_svi, sample_guide, subset_plays

from conftest import FIXTURES


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    settings = Settings(paths=Paths(root=tmp_path_factory.mktemp("pm")))
    install_fixtures(FIXTURES / "clean", settings)
    build_all(settings, (2014, 2024, 2025), log=lambda m: None)
    return build_dataset(settings)


def test_dataset_keeps_only_eligible_non_garbage_plays(ds):
    p = ds.plays
    assert p.height > 300 and set(p["season"].unique()) == {2014, 2024, 2025}
    assert p["off_ts"].null_count() == 0 and p["def_ts"].null_count() == 0
    # sacks are dropbacks with no target term; rushes have no passer
    assert not p.filter(pl.col("is_sack"))["has_target_term"].any()
    assert p.filter(pl.col("play_kind") == 1)["pass_rp"].null_count() == p.filter(pl.col("play_kind") == 1).height


def test_levels_and_random_walk():
    parent = np.array([-1, 0, 1, -1, 3])  # two chains: 0->1->2 and 3->4
    gap = np.array([0, 1, 2, 0, 1])
    lv = M._levels(parent, gap, np.arange(5))
    assert [list(x) for x in lv.idx] == [[0, 3], [1, 4], [2]]
    z = jnp.array([1.0, 1.0, 1.0, -1.0, 2.0])
    out = M._chain(z, M._levels_data(lv), first_scale=0.5, step_scale=0.1)
    # s0 = 0.5; s1 = s0 + 0.1*1; s2 = s1 + 0.1*sqrt(2)*1; s3 = -0.5; s4 = s3 + 0.2
    np.testing.assert_allclose(out, [0.5, 0.6, 0.6 + 0.1 * np.sqrt(2), -0.5, -0.3], rtol=1e-5)


def test_compact_arrays_reindex_and_cut_chains(ds):
    one = ds.plays.filter(pl.col("season") == 2025)
    a = M.make_arrays(ds, one, compact=True)
    for role in ROLES:
        r = a.roles[role]
        known = r.play_known.astype(bool)
        assert (r.play_rp[known] < r.n_rp).all() and (r.play_ps[known] < r.n_ps).all()
        assert len(r.levels.idx) == 1  # one season: every player-season is a chain root
    assert a.off_ts.max() < a.n_ts and a.ts_cg.max() < a.n_cg


def test_transfer_indicator_respects_as_of(ds):
    every = transfer_indicator(ds, "recv", None)
    early = transfer_indicator(ds, "recv", datetime(2015, 1, 1, tzinfo=timezone.utc))
    assert early.sum() <= every.sum()
    assert early.sum() == 0  # the portal data starts in the 2021 cycle


def test_short_svi_fit_runs_and_beats_nothing_on_heldout(ds):
    train = ds.plays.filter(pl.col("season") <= 2024)
    a = M.make_arrays(ds, ds.plays, compact=False, mask=(ds.plays["season"] <= 2024).to_numpy())
    st, d = M.to_device(a)
    guide, params, trace = run_svi(st, d, "normal", 300, lr=0.01, log_every=100)
    assert np.isfinite([t[1] for t in trace]).all()
    batches = sample_guide(guide, params, n=10, batch=5)
    eff = effects_from_samples(st, d, batches, ["O", "D", "pass_c", "pass_s"])
    assert eff["O"].shape == (10, st.n_ts) and eff["pass_s"].shape[1] == st.role("pass")["n_ps"]
    test = np.flatnonzero((ds.plays["season"] == 2025).to_numpy())
    st_t, d_t = subset_plays(st, d, test)
    lpd = heldout_loglik(st_t, d_t, batches)
    assert lpd.shape == (len(test),) and np.isfinite(lpd).all()
    assert train.height + len(test) == ds.plays.height


def test_fixed_scales_are_held_not_sampled(ds):
    from numpyro import handlers

    a = M.make_arrays(ds, ds.plays.filter(pl.col("season") == 2025), compact=True)
    st, d = M.to_device(a)
    d["fixed"] = {"pass_tau_c": jnp.float32(0.04), "O_tau_team0": jnp.float32(0.06)}
    tr = handlers.trace(handlers.seed(M.model, 0)).get_trace(st, d)
    assert tr["pass_tau_c"]["type"] == "deterministic" and float(tr["pass_tau_c"]["value"]) == pytest.approx(0.04)
    assert tr["O_tau_team0"]["type"] == "deterministic"
    assert tr["rush_tau_c"]["type"] == "sample" and not tr["rush_tau_c"]["is_observed"]
    assert set(M.WITHIN_SEASON_SCALES).isdisjoint(M.SEASON_STEP_SCALES)
    assert {k for k, v in tr.items() if "_tau_" in k or k.endswith(("_rho", "_fcs_scale"))} == \
        set(M.WITHIN_SEASON_SCALES) | set(M.SEASON_STEP_SCALES)


def test_hyper_guard_blocks_scales_from_the_fit_season_or_later():
    from cfb.player_model.validate import check_hyper_asof, scales_for

    check_hyper_asof((2014,), 2015, burn_in=(2014,))
    check_hyper_asof((2014,), 2014, burn_in=(2014,))  # burn-in fits are never evaluated
    with pytest.raises(ValueError):
        check_hyper_asof((2014, 2015), 2015, burn_in=(2014,))
    with pytest.raises(ValueError):
        check_hyper_asof((2019,), 2016, burn_in=(2014,))
    hyper = {y: {"pass_tau_c": float(y - 2000)} for y in range(2014, 2020)}
    assert scales_for(hyper, 2014, (2014,)) == ((2014,), {"pass_tau_c": 14.0})
    assert scales_for(hyper, 2015, (2014,)) == ((2014,), {"pass_tau_c": 14.0})
    assert scales_for(hyper, 2019, (2014,)) == ((2016, 2017, 2018), {"pass_tau_c": 17.0})
    with pytest.raises(ValueError):  # nothing from the three seasons before 2025
        scales_for(hyper, 2025, (2014,))
    with pytest.raises(ValueError):  # an evaluated season never falls back to its own fit
        scales_for({2016: {"pass_tau_c": 1.0}}, 2016, (2014,))
    with pytest.raises(ValueError, match="2016"):  # a missing season in the window is an error, not skipped
        scales_for({y: hyper[y] for y in (2014, 2015, 2017)}, 2018, (2014,))


def _write_hyper(settings, seasons):
    import json
    from cfb.player_model.validate import hyper_dir

    hyper_dir(settings).mkdir(parents=True, exist_ok=True)
    for f in hyper_dir(settings).glob("*.json"):
        f.unlink()
    for y in seasons:
        (hyper_dir(settings) / f"season={y}.json").write_text(json.dumps(
            {"season": y, "scales": {k: 0.05 if "tau" in k else 0.3 for k in M.WITHIN_SEASON_SCALES}}))


def test_asof_pipeline_end_to_end(tmp_path_factory):
    import json
    from cfb.clean.common import read_table
    from cfb.player_model.asof import asof_dir, run_asof, schedule
    from cfb.player_model.data import model_dir
    from cfb.player_model.defense import run_defense_asof

    settings = Settings(paths=Paths(root=tmp_path_factory.mktemp("asof")))
    install_fixtures(FIXTURES / "clean", settings)
    build_all(settings, (2014, 2024, 2025), log=lambda m: None)
    data = build_dataset(settings)
    data.save(model_dir(settings))
    _write_hyper(settings, (2024,))
    with pytest.raises(ValueError, match="no scale fit"):  # 2024's own scales may not inform 2024 fits
        run_asof(settings, (2024, 2025), cold_steps=60, warm_steps=20, draws=8, log=lambda m: None)
    _write_hyper(settings, (2021, 2022, 2023, 2024))
    run_asof(settings, (2024, 2025), cold_steps=60, warm_steps=20, draws=8, log=lambda m: None)
    out = asof_dir(settings)
    fits = pl.concat([pl.read_parquet(p) for p in (out / "fits").glob("*.parquet")])
    assert fits.height == len(schedule(data, read_table(settings, "games"), (2024, 2025)))
    # No fit ever sees a play from its own future.
    for row in fits.iter_rows(named=True):
        later = data.plays.filter((pl.col("season") > row["season"]) | (
            (pl.col("season") == row["season"]) & (pl.col("week_ord") > row["through_week"])))
        assert row["plays_used"] == data.plays.height - later.height
    vals = pl.concat([pl.read_parquet(p) for p in (out / "player_values").glob("*.parquet")])
    assert {"pass", "rush", "recv"} <= set(vals["role"].unique())
    assert vals["value_sd"].min() > 0 and vals.select(pl.col("value_mean").is_finite().all()).item()
    pre = vals.filter((pl.col("season") == 2025) & (pl.col("through_week") == 0))
    assert pre.height > 0 and pre["is_projection"].all() and (pre["plays_season"] == 0).all()
    from cfb.player_model.defense import EVENTS, hyper_dir as defense_hyper_dir

    defense_hyper_dir(settings).mkdir(parents=True)
    for y in (2021, 2022, 2023, 2024):
        (defense_hyper_dir(settings) / f"season={y}.json").write_text(
            json.dumps({"season": y, "scales": {"tau_c": [0.3] * len(EVENTS)}}))
    run_defense_asof(settings, schedule(data, read_table(settings, "games"), (2024, 2025)), cold_steps=60,
                     warm_steps=20, draws=8, log=lambda m: None)
    dv = pl.concat([pl.read_parquet(p) for p in (out / "defense_values").glob("*.parquet")])
    assert dv.height > 0
    assert dv.height > 0 and dv["delta_sacks"][0] < 0 < -dv["delta_ints"][0]


def test_defense_subset_reindexes_and_cuts_chains():
    from cfb.player_model.defense import DefenseData, subset_seasons

    # defender 0: 2016 -> 2017 -> 2018; defender 1: 2017 -> 2018; defender 2: 2016 only
    seasons = pl.DataFrame({"ds_idx": [0, 1, 2, 3, 4, 5], "dp_idx": [0, 0, 0, 1, 1, 2],
                            "athlete_id": [10, 10, 10, 11, 11, 12], "season": [2016, 2017, 2018, 2017, 2018, 2016],
                            "parent": [-1, 0, 1, -1, 3, -1], "gap": [0, 1, 1, 0, 1, 0]})
    rows = seasons.select("dp_idx", "ds_idx", "season")
    dd = DefenseData(rows, pl.DataFrame({"dp_idx": [0, 1, 2], "athlete_id": [10, 11, 12]}), seasons, ["DL"])
    sub = subset_seasons(dd, (2017, 2018))
    assert sub.defenders["athlete_id"].to_list() == [10, 11]
    assert sub.seasons["ds_idx"].to_list() == [0, 1, 2, 3] and sub.seasons["dp_idx"].to_list() == [0, 0, 1, 1]
    # 2017 rows become chain roots; 2018 rows point at the re-indexed 2017 rows
    assert sub.seasons["parent"].to_list() == [-1, 0, -1, 2] and sub.seasons["gap"].to_list() == [0, 1, 0, 1]
    assert sub.rows["ds_idx"].to_list() == [0, 1, 2, 3] and sub.rows["dp_idx"].to_list() == [0, 0, 1, 1]
