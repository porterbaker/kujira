"""Command-line entry point: ``cfb <group> <command>``.

Every command that would call the CFBD API is a dry run unless given ``--execute``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from urllib.parse import urlencode

from cfb.config import Settings, parse_seasons
from cfb.ingest.planner import Plan, build_plan, format_plan
from cfb.ingest.raw_store import partition_dir
from cfb.ingest.registry import BY_SLUG

EXIT_NOT_IMPLEMENTED = 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfb", description="CFB player value and game prediction")
    groups = parser.add_subparsers(dest="group", required=True, metavar="<group>")

    ingest = groups.add_parser("ingest", help="pull CFBD endpoints into data/raw")
    ingest_cmds = ingest.add_subparsers(dest="command", required=True, metavar="<command>")

    plan = ingest_cmds.add_parser("plan", help="print planned requests and call count; never calls the API")
    _add_ingest_args(plan)
    plan.set_defaults(func=cmd_ingest_plan)

    execute = ingest_cmds.add_parser("execute", help="run ingestion (dry run unless --execute)")
    _add_ingest_args(execute)
    execute.add_argument("--execute", action="store_true", help="actually call the API")
    execute.add_argument("--stage", choices=("A", "B", "all"), default="all",
                         help="A = lookups, calendar, games; B = everything planned from them (default: all)")
    execute.add_argument("--max-calls", type=int, help="abort before exceeding this many counted calls")
    execute.set_defaults(func=cmd_ingest_execute)

    summary = ingest_cmds.add_parser("summary", help="offline: write reports/phase1b_ingest.md from the manifest")
    summary.add_argument("--no-schema", action="store_true", help="skip the per-file schema check")
    summary.set_defaults(func=cmd_ingest_summary)

    probe = ingest_cmds.add_parser("probe", help="Phase 1a probe: a fixed set of live calls (dry run unless --execute)")
    probe.add_argument("--execute", action="store_true", help="actually call the API")
    probe.add_argument("--report", action="store_true",
                       help="offline: write reports/phase1a_probe.md and tests/fixtures from landed probe data")
    probe.set_defaults(func=cmd_ingest_probe)

    clean = groups.add_parser("clean", help="raw JSON to typed parquet tables")
    clean_cmds = clean.add_subparsers(dest="command", required=True, metavar="<command>")
    clean_build = clean_cmds.add_parser("build", help="build data/clean tables from data/raw (offline)")
    clean_build.add_argument("--seasons", help="default: config range")
    clean_build.set_defaults(func=cmd_clean_build)
    clean_cmds.add_parser("report", help="write reports/phase2_data_quality.md (offline)").set_defaults(
        func=cmd_clean_report)

    player = groups.add_parser("player", help="stage 1 player value model (offline)")
    pcmds = player.add_subparsers(dest="command", required=True, metavar="<command>")
    pcmds.add_parser("dataset", help="build data/model from data/clean").set_defaults(func=cmd_player_dataset)
    pv = pcmds.add_parser("validate", help="NUTS diagnostics and SVI vs NUTS on one or more seasons")
    pv.add_argument("--seasons", default="2019", help="e.g. 2019 or 2014-2015 (default 2019)")
    pv.add_argument("--chains", type=int, default=4)
    pv.add_argument("--warmup", type=int, default=500)
    pv.add_argument("--samples", type=int, default=500)
    pv.add_argument("--svi-steps", type=int, default=30000)
    pv.set_defaults(func=cmd_player_validate)
    ph = pcmds.add_parser("hyper", help="single-season NUTS scale fits for the as-of fits; resumable")
    ph.add_argument("--seasons", help="default: every config season but the last (the scales of season S's fits "
                                      "come from the seasons before S)")
    ph.add_argument("--chains", type=int, default=4)
    ph.add_argument("--warmup", type=int, default=500)
    ph.add_argument("--samples", type=int, default=500)
    ph.add_argument("--skip-defense", action="store_true", help="offensive scales only")
    ph.add_argument("--defense-only", action="store_true", help="defender scales only")
    ph.set_defaults(func=cmd_player_hyper)
    pvd = pcmds.add_parser("validate-defense", help="defender model: NUTS vs SVI on one season; writes its scales")
    pvd.add_argument("--season", type=int, default=2016)
    pvd.set_defaults(func=cmd_player_validate_defense)
    pf = pcmds.add_parser("fit", help="as-of weekly fits (offense + defenders); resumable")
    pf.add_argument("--seasons", help="default: config range")
    pf.add_argument("--cadence", type=int, default=1, help="refit every N weeks (default 1)")
    pf.add_argument("--cold-steps", type=int, default=10000)
    pf.add_argument("--warm-steps", type=int, default=600)
    pf.add_argument("--skip-defense", action="store_true")
    pf.set_defaults(func=cmd_player_fit)
    pcmds.add_parser("report", help="write reports/phase3_player_model.md").set_defaults(func=cmd_player_report)

    game = groups.add_parser("game", help="stage 2 game model")
    game.add_subparsers(dest="command", required=True, metavar="<command>").add_parser(
        "fit", help="fit the game margin model"
    ).set_defaults(func=_not_implemented("game fit", "Phase 4"))

    groups.add_parser("backtest", help="rolling-origin backtest").set_defaults(
        func=_not_implemented("backtest", "Phase 5")
    )
    groups.add_parser("report", help="write evaluation reports").set_defaults(
        func=_not_implemented("report", "Phase 5")
    )
    return parser


def _add_ingest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seasons", help="e.g. 2024, 2021-2025, or 2019,2021-2023 (default: config range)")
    parser.add_argument("--endpoints", help="comma-separated endpoint slugs (default: all), e.g. plays,plays_stats")
    parser.add_argument("--fcs", action="store_true", help="include FCS-only games (default: FBS scope)")
    parser.add_argument("--weather", action="store_true", help="include /games/weather")
    parser.add_argument("--refresh", action="store_true", help="re-pull current-season partitions already on disk")
    parser.add_argument("--list", action="store_true", help="also list every planned request")


def _plan_from_args(args: argparse.Namespace) -> tuple[Settings, Plan]:
    base = Settings()
    settings = replace(base, ingest=replace(base.ingest, include_fcs=args.fcs, include_weather=args.weather))
    seasons = parse_seasons(args.seasons, settings.ingest.current_season) if args.seasons else settings.seasons
    return settings, build_plan(seasons, settings, endpoints=_endpoints(args), refresh=args.refresh)


def _endpoints(args: argparse.Namespace) -> set[str] | None:
    if not args.endpoints:
        return None
    endpoints = {e.strip() for e in args.endpoints.split(",") if e.strip()}
    unknown = endpoints - BY_SLUG.keys()
    if unknown:
        raise SystemExit(f"unknown endpoint(s): {', '.join(sorted(unknown))}; known: {', '.join(BY_SLUG)}")
    return endpoints


def _print_plan(settings: Settings, plan: Plan, list_requests: bool) -> None:
    print(format_plan(plan))
    if list_requests:
        print()
        for req in plan.requests:
            target = partition_dir(settings.paths.raw, req.spec.slug, req.partition).relative_to(settings.paths.root)
            query = f"?{urlencode(req.params)}" if req.params else ""
            print(f"GET {req.spec.path}{query}  ->  {target.as_posix()}")
        for block in plan.estimates:
            print(f"~{block.count:>4} x GET {block.spec.path}  (season {block.season}; {block.reason})")


def cmd_ingest_plan(args: argparse.Namespace) -> int:
    settings, plan = _plan_from_args(args)
    _print_plan(settings, plan, args.list)
    print("\nDry run: no requests sent.")
    return 0


def cmd_ingest_execute(args: argparse.Namespace) -> int:
    settings, plan = _plan_from_args(args)
    _print_plan(settings, plan, args.list)
    if not args.execute:
        print("\nDry run: no requests sent. Re-run with --execute to call the API.")
        return 0

    from cfb.api.client import CFBDClient
    from cfb.config import load_api_key
    from cfb.ingest import bulk
    from cfb.ingest.manifest import Manifest

    stages = ("A", "B") if args.stage == "all" else (args.stage,)
    print()
    with CFBDClient(load_api_key(settings.paths), settings.api) as client:
        try:
            result = bulk.run(settings, client, Manifest(settings.paths), plan.seasons, endpoints=_endpoints(args),
                              stages=stages, refresh=args.refresh, max_calls=args.max_calls,
                              log=lambda msg: print(msg, flush=True))
        except bulk.Abort as exc:
            print(f"ABORTED: {exc}. Landed partitions are kept; re-run to resume.", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("Interrupted. Landed partitions are kept; re-run to resume.", file=sys.stderr)
            return 130
    print(f"done: {result.requests} requests, {result.data_calls} counted calls, {result.empty} empty responses, "
          f"{len(result.truncated)} truncated pages, {len(result.failures)} failures")
    for slug, params, status in result.failures:
        print(f"  failed: {slug} {params} HTTP {status}")
    return 0 if not result.failures else 1


def _log(msg: str) -> None:
    print(msg, flush=True)


def cmd_player_dataset(args: argparse.Namespace) -> int:
    from cfb.player_model.data import build_dataset, model_dir

    settings = Settings()
    ds = build_dataset(settings)
    ds.save(model_dir(settings))
    print(f"wrote {model_dir(settings).relative_to(settings.paths.root)}: {ds.plays.height:,} plays")
    return 0


def cmd_player_validate(args: argparse.Namespace) -> int:
    from cfb.player_model.validate import run_validation

    settings = Settings()
    seasons = parse_seasons(args.seasons, settings.ingest.current_season)
    run_validation(settings, seasons, args.chains, args.warmup, args.samples, args.svi_steps, log=_log)
    return 0


def cmd_player_validate_defense(args: argparse.Namespace) -> int:
    from cfb.player_model.defense import validate_defense

    validate_defense(Settings(), args.season, log=_log)
    return 0


def cmd_player_hyper(args: argparse.Namespace) -> int:
    from cfb.player_model.defense import fit_defense_scales
    from cfb.player_model.validate import fit_hyper

    settings = Settings()
    seasons = (parse_seasons(args.seasons, settings.ingest.current_season) if args.seasons
               else settings.seasons[:-1])
    if not args.defense_only:
        fit_hyper(settings, seasons, args.chains, args.warmup, args.samples, log=_log)
    if not args.skip_defense:
        fit_defense_scales(settings, seasons, args.chains, args.warmup, args.samples, log=_log)
    return 0


def cmd_player_fit(args: argparse.Namespace) -> int:
    from cfb.clean.common import read_table
    from cfb.player_model.asof import run_asof, schedule
    from cfb.player_model.data import Dataset, model_dir
    from cfb.player_model.defense import run_defense_asof

    settings = Settings()
    seasons = parse_seasons(args.seasons, settings.ingest.current_season) if args.seasons else settings.seasons
    run_asof(settings, seasons, cadence=args.cadence, cold_steps=args.cold_steps, warm_steps=args.warm_steps, log=_log)
    if not args.skip_defense:
        ds = Dataset.load(model_dir(settings))
        plan = schedule(ds, read_table(settings, "games"), seasons, args.cadence)
        del ds
        run_defense_asof(settings, plan, log=_log)
    return 0


def cmd_player_report(args: argparse.Namespace) -> int:
    from cfb.player_model.report import write_report

    settings = Settings()
    print("wrote", write_report(settings).relative_to(settings.paths.root))
    return 0


def cmd_clean_build(args: argparse.Namespace) -> int:
    from cfb.clean.build import build_all

    settings = Settings()
    seasons = parse_seasons(args.seasons, settings.ingest.current_season) if args.seasons else settings.seasons
    build_all(settings, seasons, log=lambda msg: print(msg, flush=True))
    return 0


def cmd_clean_report(args: argparse.Namespace) -> int:
    from cfb.clean.quality import write_report

    settings = Settings()
    print("wrote", write_report(settings).relative_to(settings.paths.root))
    return 0


def cmd_ingest_summary(args: argparse.Namespace) -> int:
    from cfb.ingest import summary

    settings = Settings()
    print("wrote", summary.write_summary(settings, check_schemas=not args.no_schema).relative_to(settings.paths.root))
    return 0


def cmd_ingest_probe(args: argparse.Namespace) -> int:
    from cfb.ingest import probe

    settings = Settings()
    if args.report:
        from cfb.ingest import probe_report

        print("wrote", probe_report.write_report(settings).relative_to(settings.paths.root))
        fixtures = probe_report.write_fixtures(settings, settings.paths.root / "tests" / "fixtures")
        print(f"wrote {len(fixtures)} fixtures to tests/fixtures/")
        return 0
    print(probe.format_probe_plan(settings))
    if not args.execute:
        print("\nDry run: no requests sent. Re-run with --execute to call the API.")
        return 0

    from cfb.api.client import CFBDClient
    from cfb.config import load_api_key
    from cfb.ingest.manifest import Manifest

    print()
    with CFBDClient(load_api_key(settings.paths), settings.api) as client:
        probe.run_probe(settings, client, Manifest(settings.paths))
    return 0


def _not_implemented(command: str, phase: str):
    def run(args: argparse.Namespace) -> int:
        print(f"cfb {command}: not implemented yet ({phase})", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED

    return run


if __name__ == "__main__":
    sys.exit(main())
