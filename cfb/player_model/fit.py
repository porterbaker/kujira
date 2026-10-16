"""Inference helpers: SVI with explicit data arguments, guide sampling, held-out log-likelihood."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal, AutoNormal
from numpyro.infer.initialization import init_to_median
from numpyro.infer.util import log_likelihood
from numpyro.optim import Adam

from cfb.player_model.model import Static, effects_model, model

PLAY_KEYS = ("y", "kind", "season", "ctx", "off_ts", "def_ts", "target_term", "mask")
ROLE_PLAY_KEYS = ("play_rp", "play_ps", "play_pg", "play_pos", "play_known")


def make_guide(kind: str, model_fn: Callable, rank: int = 16):
    if kind == "normal":
        return AutoNormal(model_fn, init_loc_fn=init_to_median(num_samples=15), init_scale=0.05)
    if kind == "lowrank":
        return AutoLowRankMultivariateNormal(model_fn, init_loc_fn=init_to_median(num_samples=15),
                                             init_scale=0.05, rank=rank)
    raise ValueError(kind)


def run_svi(st: Static, d: dict, guide_kind: str, steps: int, lr: float = 0.01, *, seed: int = 0,
            init_params: dict | None = None, model_fn=model, log: Callable[[str], None] | None = None,
            log_every: int = 1000) -> tuple[object, dict, list[tuple[int, float]]]:
    """Fit an autoguide; returns (guide, params, [(step, mean loss over the last log_every steps)]).
    `init_params` (constrained guide params from a fit with the same shapes) warm-starts the fit."""
    m = functools.partial(model_fn, st)
    guide = make_guide(guide_kind, m)
    svi = SVI(m, guide, Adam(lr), Trace_ELBO())
    state = svi.init(jax.random.PRNGKey(seed), d, init_params=init_params)
    step = jax.jit(svi.update)
    trace, acc, started = [], jnp.zeros(()), time.monotonic()
    for i in range(1, steps + 1):
        state, loss = step(state, d)
        acc = acc + loss
        if i % log_every == 0 or i == steps:
            n = log_every if i % log_every == 0 else i % log_every
            trace.append((i, float(acc) / n))
            acc = jnp.zeros(())
            if log is not None:
                log(f"    step {i:>6}/{steps}  mean ELBO loss {trace[-1][1]:,.0f}  ({time.monotonic() - started:.0f}s)")
    return guide, svi.get_params(state), trace


def run_chains(model_fn: Callable, d: dict, chains: int, warmup: int, samples: int, keep: set[str],
               seed: int = 0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """NUTS one chain at a time, keeping only the `keep` sites. Peak memory is one chain's draws of every
    latent plus the kept sites of all chains, not every site of every chain; deterministic sites outside
    `keep` are never computed. Chains start at prior medians (effects at 0): numpyro's default, uniform(-2, 2) on
    the unconstrained scale, starts the scales at 0.1-7 against true values of 0.01-0.3, and about one run in two
    then had a chain that adapted a tiny step size and never left its start (2015, 2018, 2014-15 at seed 0).
    Returns ({site: (chain, draw, ...)}, {"diverging", "num_steps": (chain, draw)})."""
    from numpyro import handlers
    from numpyro.infer import MCMC, NUTS

    m = handlers.block(model_fn, hide_fn=lambda site: site["type"] == "deterministic" and site["name"] not in keep)
    draws: dict[str, list] = {}
    extra: dict[str, list] = {"diverging": [], "num_steps": []}
    for key in jax.random.split(jax.random.PRNGKey(seed), chains):
        # jit_model_args: the data are arguments of the compiled sampler, not constants baked into it; each
        # chain's executable would otherwise hold its own copy of the data in JAX's cache.
        kernel = NUTS(m, target_accept_prob=0.85, max_tree_depth=8, init_strategy=init_to_median(num_samples=15))
        mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples,
                    num_chains=1, progress_bar=False, jit_model_args=True)
        mcmc.run(key, d, extra_fields=tuple(extra))
        for k, v in mcmc.get_samples().items():
            if k in keep:
                draws.setdefault(k, []).append(np.asarray(v))
        for k, v in mcmc.get_extra_fields().items():
            extra[k].append(np.asarray(v))
        del mcmc
        jax.clear_caches()
    return {k: np.stack(v) for k, v in draws.items()}, {k: np.stack(v) for k, v in extra.items()}


def sample_guide(guide, params: dict, n: int, seed: int = 1, batch: int = 25) -> list[dict]:
    """Posterior draws of every latent site, in batches to bound memory."""
    out, key = [], jax.random.PRNGKey(seed)
    for start in range(0, n, batch):
        key, sub = jax.random.split(key)
        out.append(guide.sample_posterior(sub, params, sample_shape=(min(batch, n - start),)))
    return out


def effects_from_samples(st: Static, d: dict, batches: list[dict], sites: list[str]) -> dict[str, np.ndarray]:
    """Run the effects-only model on posterior draws; returns sites stacked as (draws, ...)."""
    got: dict[str, list] = {s: [] for s in sites}
    for i, b in enumerate(batches):
        keep = {k: v for k, v in b.items() if not (k.endswith("_g_z") or (k.endswith("_g") and "_tau_" not in k))}
        res = Predictive(functools.partial(effects_model, st), posterior_samples=keep, batch_ndims=1,
                         return_sites=sites)(jax.random.PRNGKey(i), d)
        for s in sites:
            got[s].append(np.asarray(res[s]))
    return {s: np.concatenate(v, axis=0) for s, v in got.items()}


def subset_plays(st: Static, d: dict, idx: np.ndarray) -> tuple[Static, dict]:
    """The same entities, restricted to the plays in `idx` (for held-out evaluation)."""
    idx = jnp.asarray(idx)
    d2 = {**d, **{k: d[k][idx] for k in PLAY_KEYS}}
    d2["mask"] = jnp.ones(len(idx), dtype=bool)
    d2["roles"] = {r: {**v, **{k: v[k][idx] for k in ROLE_PLAY_KEYS}} for r, v in d["roles"].items()}
    st2 = Static(**{**st.__dict__, "n_plays": int(len(idx))})
    return st2, d2


def heldout_loglik(st: Static, d: dict, batches: list[dict], model_fn=model) -> np.ndarray:
    """Pointwise log predictive density: log mean_draws p(y_i | draw), one value per play."""
    lls = []
    for b in batches:
        ll = log_likelihood(functools.partial(model_fn, st), b, d, batch_ndims=1)["y"]
        lls.append(np.asarray(ll))
    ll = np.concatenate(lls, axis=0)
    return np.asarray(jax.scipy.special.logsumexp(ll, axis=0) - np.log(ll.shape[0]))
