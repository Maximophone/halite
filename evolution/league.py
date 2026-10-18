#!/usr/bin/env python3
"""V2: evolutionary self-play league, larger maps and fresh champion promotion gates."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

from train import HERE, ROOT, FEATURES, atomic_json, evaluate, export_report, tuple_tree

EXTRA_FEATURES = ["Stay × strength", "Stay × production", "Stay × strength²", "Stay × nearby enemies",
                  "Stay × nearby allies", "Friendly × merged strength", "Neutral × strength margin",
                  "Neutral × strength deficit", "Enemy × strength margin", "Strength × nearby enemies",
                  "Frontier progress × strength", "Frontier progress × depth", "Enemy progress × strength",
                  "Neutral production / resistance", "Friendly target strength", "Stay × strength / production"]
PARAMS = 248


def read_policy(path):
    tokens = path.read_text().split()
    if tokens[0] not in ("HALITE_EVO_V1", "HALITE_EVO_V2"):
        raise ValueError("Only evolved V1/V2 policy checkpoints are supported")
    weights = list(map(float, tokens[1:]))
    expected = 232 if tokens[0].endswith("V1") else PARAMS
    if len(weights) != expected or not all(math.isfinite(w) for w in weights):
        raise ValueError("Invalid checkpoint weights")
    return weights + [0.0] * (PARAMS - len(weights))


def write_policy(path, weights):
    if len(weights) != PARAMS or not all(math.isfinite(w) for w in weights):
        raise ValueError("Expected 248 finite weights")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("HALITE_EVO_V2\n" + "\n".join(format(w, ".9g") for w in weights) + "\n")


def mutate(rng, parent, ident, scale=None):
    sigma = scale or min(.5, max(.025, parent["sigma"] * math.exp(rng.gauss(0, .18))))
    weights = parent["weights"][:]
    block = rng.random()
    for j in range(PARAMS):
        # Search the new conditional features efficiently without freezing the inherited net.
        active = (j >= 232) if block < .45 else (j < 232 and rng.random() < .12) if block < .85 else rng.random() < .25
        if active:
            weights[j] = max(-8, min(8, weights[j] + rng.gauss(0, sigma)))
    return {"id": ident, "parent": parent["id"], "sigma": sigma, "weights": weights}


def mean_score(rows):
    return statistics.mean(row["score"] for row in rows)


def batch(paths, opponents, scenarios):
    return [(path, opponent, seed, size, seat) for path in paths for opponent in opponents
            for seed, size in scenarios for seat in (0, 1)]


def run(args):
    directory = args.run.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another trainer is writing this run")
        train(args, directory)


def train(args, run):
    rng = random.Random(args.seed)
    if args.resume:
        state = json.loads((run / "state.json").read_text())
        if state.get("version") != 2:
            raise SystemExit("--resume requires a V2 league run")
        rng.setstate(tuple_tree(state["rng"]))
        config = state["config"]
    else:
        if (run / "state.json").exists():
            raise SystemExit("Run exists: use --resume or another directory")
        initial = []
        sources = []
        for i, source in enumerate(args.seed_policies):
            weights = read_policy(source)
            ident = f"seed-{i}"
            path = run / "initial" / f"{ident}.evo"
            write_policy(path, weights)
            initial.append({"id": ident, "parent": None, "sigma": .12, "weights": weights, "path": str(path)})
            sources.append({"path": str(source.resolve()), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        config = {"seed": args.seed, "population": args.population, "opponents": args.opponents, "maps": args.maps,
                  "sizes": args.sizes, "gate_maps": args.gate_maps, "initialization": "expanded self-play-evolved ancestors",
                  "algorithm": "elitist mutation-only league with promotion gates", "sources": sources,
                  "fitness": "self-play wins only; draw = 0.5", "version": 2}
        population = initial[:args.population]
        while len(population) < args.population:
            population.append(mutate(rng, rng.choice(initial), f"g0000-{len(population):03d}", rng.choice([.08, .16, .3])))
        state = {"version": 2, "config": config, "population": population, "generation": -1, "history": [],
                 "features": FEATURES + EXTRA_FEATURES, "params": PARAMS, "hidden": 8, "champion": initial[0]["path"],
                 "incumbent": initial[0], "archive": [p["path"] for p in initial], "anchors": [p["path"] for p in initial],
                 "recent": [p["path"] for p in initial], "games": 0, "promotions": 0, "rng": rng.getstate(),
                 "updated": time.time(), "status": "training"}
        atomic_json(run / "state.json", state)
    if state["generation"] + 1 >= args.generations:
        write_policy(run / "champion.evo", state["incumbent"]["weights"])
        if args.export:
            export_report(run, args.export.resolve())
        return
    state["status"] = "training"
    atomic_json(run / "state.json", state)
    for generation in range(state["generation"] + 1, args.generations):
        tick = time.monotonic()
        directory = run / f"gen-{generation:04d}"
        directory.mkdir(exist_ok=True)
        population = state["population"]
        paths = []
        for i, p in enumerate(population):
            path = directory / f"{i:03d}.evo"
            write_policy(path, p["weights"])
            paths.append(str(path))
        pool = list(dict.fromkeys(state["recent"] + state["archive"]))
        panel = list(dict.fromkeys([state["champion"], rng.choice(state["anchors"])]))
        candidates = [p for p in pool if p not in panel]
        rng.shuffle(candidates)
        panel += candidates[:max(0, config["opponents"] - len(panel))]
        while len(panel) < config["opponents"]:
            panel.append(paths[len(panel) % len(paths)])
        panel = panel[:config["opponents"]]
        scenarios = [(rng.randrange(1, 1_000_000_000), rng.choice(config["sizes"])) for _ in range(config["maps"])]
        jobs = batch(paths, panel, scenarios)
        rows = evaluate(jobs, directory, args.workers)
        per = len(rows) // len(paths)
        scores = [mean_score(rows[i*per:(i+1)*per]) for i in range(len(paths))]
        order = list(range(len(paths)))
        rng.shuffle(order)
        order.sort(key=lambda i: scores[i], reverse=True)
        contender = population[order[0]]
        contender_path = paths[order[0]]
        # Compare contender and incumbent on exactly the same fresh league matches.
        gate_pool = list(dict.fromkeys([state["champion"], state["anchors"][0]] + state["recent"][:2] + panel))[:6]
        gate_scenarios = [(rng.randrange(1_000_000_000, 2_000_000_000), rng.choice(config["sizes"])) for _ in range(config["gate_maps"])]
        gate_jobs = batch([contender_path, state["champion"]], gate_pool, gate_scenarios)
        gate_rows = evaluate(gate_jobs, directory, args.workers)
        half = len(gate_rows) // 2
        challenger_score, incumbent_score = mean_score(gate_rows[:half]), mean_score(gate_rows[half:])
        promoted = challenger_score > incumbent_score + .025
        if promoted:
            state["incumbent"] = {**contender, "path": contender_path}
            state["champion"] = contender_path
            state["archive"].append(contender_path)
            state["promotions"] += 1
        champion = state["incumbent"]
        # Fixed evaluation never feeds selection. Full map-size range, separate seed namespace.
        eval_jobs = batch([state["champion"]], state["anchors"], [(3_000_000_001+k, size) for k, size in enumerate((20, 32, 40, 50))])
        eval_rows = evaluate(eval_jobs, directory, args.workers)
        replay_path = directory / "replay.json"
        evaluate([(state["champion"], state["anchors"][0], 2_700_000_001, 32, 0)], directory, 1, replay_path)
        means = [statistics.mean(p["weights"][j] for p in population) for j in range(PARAMS)]
        diversity = math.sqrt(statistics.mean((p["weights"][j]-means[j])**2 for p in population for j in range(PARAMS)))
        telemetry = [{"id": p["id"], "parent": p["parent"], "score": scores[i], "sigma": p["sigma"],
                      "coverage": statistics.mean(r["coverage"] for r in rows[i*per:(i+1)*per]),
                      "move_rate": statistics.mean(r["move_rate"] for r in rows[i*per:(i+1)*per])}
                     for i, p in enumerate(population)]
        best = telemetry[order[0]]
        state["history"].append({"generation": generation, "best": max(scores), "mean": statistics.mean(scores),
            "worst": min(scores), "anchor_score": mean_score(eval_rows), "anchor_games": len(eval_rows),
            "diversity": diversity, "sigma": statistics.mean(p["sigma"] for p in population),
            "seconds": time.monotonic()-tick, "games": len(rows), "population": telemetry,
            "champion_id": champion["id"], "champion_parent": champion["parent"], "weights": champion["weights"],
            "display_weights": champion["weights"][208:], "scenarios": scenarios,
            "coverage": best["coverage"], "move_rate": best["move_rate"],
            "gate_challenger": challenger_score, "gate_incumbent": incumbent_score, "gate_games": half,
            "promoted": promoted, "promotions": state["promotions"], "contender_id": contender["id"]})
        # Preserve more elites and vary the scope and scale of weight mutation.
        keep = max(2, config["population"] // 8)
        parents = [population[i] for i in order[:max(4, config["population"] // 3)]]
        next_population = [dict(champion)]
        next_population += [dict(population[i]) for i in order if population[i]["id"] != champion["id"]][:keep-1]
        while len(next_population) < config["population"]:
            parent = rng.choice(parents + [champion]*3)
            scale = rng.choice([.04, .08, .16, .32]) if rng.random() < .15 else None
            next_population.append(mutate(rng, parent, f"g{generation+1:04d}-{len(next_population):03d}", scale))
        state.update(generation=generation, population=next_population, recent=[paths[i] for i in order[:8]],
                     games=state["games"]+len(jobs)+len(gate_jobs)+len(eval_jobs)+1,
                     replay=str(replay_path.relative_to(run)), updated=time.time(), rng=rng.getstate(),
                     status="complete" if generation+1 == args.generations else "training")
        atomic_json(run / "state.json", state)
        write_policy(run / "champion.evo", champion["weights"])
        print(f"gen {generation:03d} | train {max(scores):.1%} | gate {challenger_score:.1%} vs {incumbent_score:.1%}"
              f" {'PROMOTED' if promoted else 'retained'} | fixed {mean_score(eval_rows):.1%} | {time.monotonic()-tick:.1f}s", flush=True)
    if args.export:
        export_report(run, args.export.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "runs/beta")
    parser.add_argument("--seed-policies", type=Path, nargs="+", default=[ROOT / "starter-kit/rust/weights/evo_bot.evo"])
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--opponents", type=int, default=6)
    parser.add_argument("--maps", type=int, default=3)
    parser.add_argument("--gate-maps", type=int, default=6)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 20, 24, 32, 40, 50])
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if min(args.generations, args.maps, args.gate_maps, args.workers) < 1 or args.population < 8 or not 2 <= args.opponents <= args.population:
        parser.error("positive counts, population >= 8, and 2 <= opponents <= population required")
    if any(s < 6 or s > 100 for s in args.sizes) or len(args.seed_policies) > args.population:
        parser.error("sizes must be 6..100 and seed policies must fit the population")
    subprocess.run(["cargo", "build", "--release", "--manifest-path", str(HERE / "Cargo.toml")], check=True)
    run(args)


if __name__ == "__main__":
    main()
