#!/usr/bin/env python3
"""Population neuroevolution from scratch. Python and Rust standard libraries only."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVALUATOR = HERE / "target/release/evo-eval"
INPUTS, HIDDEN = 24, 8
PARAMS = INPUTS * HIDDEN + 2 * HIDDEN + INPUTS
FEATURES = ["Source strength", "Source production", "Target strength", "Target production",
            "Friendly target", "Neutral target", "Enemy target", "Stay still", "Strength difference",
            "Nearby friendly strength", "Nearby enemy strength", "Friendly neighbors", "Enemy neighbors",
            "Neutral neighbors", "Distance to frontier", "Progress toward frontier", "Distance to enemy",
            "Progress toward enemy", "Elapsed game", "Territory share", "Production balance",
            "Potential overflow", "Two-step strength", "Two-step production"]


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False))
    temp.replace(path)


def write_policy(path, weights):
    assert len(weights) == PARAMS and all(math.isfinite(w) for w in weights)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("HALITE_EVO_V1\n" + "\n".join(format(w, ".9g") for w in weights) + "\n")


def fresh(rng, ident):
    return {"id": ident, "parent": None, "sigma": 0.12,
            "weights": [rng.gauss(0, 0.4) for _ in range(PARAMS)]}


def reproduce(rng, ranked, generation, size):
    elites = ranked[:max(2, size // 4)]
    population = [dict(p) for p in ranked[:2]]
    for i in range(2, size):
        ident = f"g{generation:04d}-{i:03d}"
        if rng.random() < 0.06:
            population.append(fresh(rng, ident))
            continue
        parent = rng.choice(elites)
        sigma = min(0.6, max(0.015, parent["sigma"] * math.exp(rng.gauss(0, 0.22))))
        dense = rng.random() < 0.2
        weights = [max(-8, min(8, w + rng.gauss(0, sigma))) if dense or rng.random() < 0.3 else w
                   for w in parent["weights"]]
        population.append({"id": ident, "parent": parent["id"], "sigma": sigma, "weights": weights})
    return population


def evaluate(jobs, directory, workers, replay=None):
    spec = directory / ("replay-jobs.tsv" if replay else "jobs.tsv")
    spec.write_text("".join("\t".join(map(str, row)) + "\n" for row in jobs))
    command = [str(EVALUATOR), str(spec), str(workers)]
    if replay:
        command.append(str(replay))
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    if len(rows) != len(jobs):
        raise RuntimeError("evaluator returned an incomplete batch")
    return rows


def tuple_tree(value):
    return tuple(tuple_tree(v) for v in value) if isinstance(value, list) else value


def snapshot(state):
    return {"version": state.get("version", 1), "config": state["config"], "generation": state["generation"],
            "games": state["games"], "history": state["history"], "features": state.get("features", FEATURES),
            "params": state.get("params", PARAMS), "hidden": HIDDEN, "champion": state["champion"],
            "updated": state["updated"], "status": state.get("status", "training")}


def export_report(run, destination):
    state = json.loads((run / "state.json").read_text())
    replay = json.loads((run / state["replay"]).read_text())
    # All embedded data originates in numeric training results and local metadata.
    payload = json.dumps({"data": snapshot(state), "replay": replay}).replace("<", "\\u003c")
    html = (HERE / "dashboard.html").read_text().replace("/* EMBED_REPORT */", "window.EVO_REPORT=" + payload + ";")
    destination.write_text(html)


def train(args):
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another trainer is writing {run}")
        train_locked(args, run)


def train_locked(args, run):
    rng = random.Random(args.seed)
    if args.resume:
        state = json.loads((run / "state.json").read_text())
        rng.setstate(tuple_tree(state["rng"]))
        config = state["config"]
        print(f"Resuming after generation {state['generation']}; using the saved training configuration.", flush=True)
    else:
        if (run / "state.json").exists():
            raise SystemExit("Run already exists; use --resume or a new --run directory")
        config = {"seed": args.seed, "population": args.population, "opponents": args.opponents,
                  "maps": args.maps, "sizes": args.sizes, "algorithm": "elitist mutation-only neuroevolution",
                  "fitness": "self-play wins (draw = 0.5), seat-paired", "initialization": "independent random weights"}
        population = [fresh(rng, f"g0000-{i:03d}") for i in range(args.population)]
        # Generation zero has a frozen common panel of randomly initialized policies.
        initial = run / "initial"
        initial.mkdir(exist_ok=True)
        opponents = []
        for i, p in enumerate(population[:args.opponents]):
            path = initial / f"{i}.evo"
            write_policy(path, p["weights"])
            opponents.append(str(path))
        state = {"generation": -1, "config": config, "population": population, "panel": opponents,
                 "archive": [], "anchors": [], "history": [], "games": 0, "champion": None,
                 "rng": rng.getstate(), "updated": time.time()}
        # Also checkpoint before the first evaluation so interrupted starts resume exactly.
        atomic_json(run / "state.json", state)
    if state["generation"] + 1 >= args.generations:
        write_policy(run / "champion.evo", state["history"][-1]["weights"])
        if args.export:
            export_report(run, args.export.resolve())
        print("Requested generation count already completed.")
        return
    state["status"] = "training"
    atomic_json(run / "state.json", state)
    start = time.monotonic()
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
        # The same maps, seats and frozen opponents for every candidate in a generation.
        scenarios = [(rng.randrange(1, 2**31), rng.choice(config["sizes"])) for _ in range(config["maps"])]
        jobs = [(path, opponent, seed, size, swap) for path in paths for opponent in state["panel"]
                for seed, size in scenarios for swap in (0, 1)]
        rows = evaluate(jobs, directory, args.workers)
        per = len(rows) // len(population)
        scores = [statistics.mean(r["score"] for r in rows[i * per:(i + 1) * per]) for i in range(len(population))]
        # Random tie breaks; territory and other telemetry NEVER affect reproduction.
        order = list(range(len(population)))
        rng.shuffle(order)
        order.sort(key=lambda i: scores[i], reverse=True)
        ranked = [population[i] for i in order]
        champion_idx = order[0]
        champion_path = paths[champion_idx]
        if not state["anchors"]:
            state["anchors"] = [paths[i] for i in order[:3]]
        # Fixed random-initialization champions, unseen seeds, and larger maps.
        # These evaluation scores never influence selection or mutation.
        eval_jobs = [(champion_path, opponent, 3_000_000_001 + k, size, swap)
                     for opponent in state["anchors"] for k, size in enumerate((16, 24, 32)) for swap in (0, 1)]
        eval_rows = evaluate(eval_jobs, directory, args.workers)
        anchor_score = statistics.mean(r["score"] for r in eval_rows)
        previous = state["archive"][-1] if state["archive"] else state["panel"][0]
        replay_path = directory / "replay.json"
        evaluate([(champion_path, previous, 2_700_000_001, 24, 0)], directory, 1, replay_path)
        mean_weights = [statistics.mean(p["weights"][j] for p in population) for j in range(PARAMS)]
        diversity = math.sqrt(statistics.mean((p["weights"][j] - mean_weights[j]) ** 2
                                             for p in population for j in range(PARAMS)))
        telemetry = [{"id": p["id"], "parent": p["parent"], "score": scores[i], "sigma": p["sigma"],
                      "coverage": statistics.mean(r["coverage"] for r in rows[i * per:(i + 1) * per]),
                      "move_rate": statistics.mean(r["move_rate"] for r in rows[i * per:(i + 1) * per])}
                     for i, p in enumerate(population)]
        entry = {"generation": generation, "best": max(scores), "mean": statistics.mean(scores),
                 "worst": min(scores), "anchor_score": anchor_score, "anchor_games": len(eval_rows),
                 "diversity": diversity, "sigma": statistics.mean(p["sigma"] for p in population),
                 "seconds": time.monotonic() - tick, "games": len(rows), "population": telemetry,
                 "champion_id": ranked[0]["id"], "champion_parent": ranked[0]["parent"],
                 "weights": ranked[0]["weights"], "scenarios": scenarios,
                 "coverage": telemetry[champion_idx]["coverage"], "move_rate": telemetry[champion_idx]["move_rate"]}
        state["history"].append(entry)
        state["archive"].append(champion_path)
        # Latest champion + historical champion + diverse recent policies. All are self-play products.
        pool = state["archive"][:-1] or state["archive"]
        panel = [champion_path]
        if config["opponents"] > 1:
            panel.append(rng.choice(pool))
        while len(panel) < config["opponents"]:
            panel.append(rng.choice(paths))
        state.update(generation=generation, population=reproduce(rng, ranked, generation + 1, config["population"]),
                     panel=panel, champion=champion_path, replay=str(replay_path.relative_to(run)),
                     games=state["games"] + len(rows) + len(eval_rows) + 1, updated=time.time())
        state["rng"] = rng.getstate()
        state["status"] = "complete" if generation + 1 == args.generations else "training"
        # State is the authoritative atomic commit; derived dashboard files can always be rebuilt.
        atomic_json(run / "state.json", state)
        write_policy(run / "champion.evo", ranked[0]["weights"])
        print(f"gen {generation:03d} | self-play {max(scores):.1%} / mean {statistics.mean(scores):.1%}"
              f" | fixed panel {anchor_score:.1%} | diversity {diversity:.3f}"
              f" | {len(rows)} games in {entry['seconds']:.1f}s", flush=True)
    print(f"Completed in {time.monotonic() - start:.1f}s. Champion: {run / 'champion.evo'}", flush=True)
    if args.export:
        export_report(run, args.export.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "runs/first")
    parser.add_argument("--generations", type=int, default=100, help="total generations, including those already saved")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--opponents", type=int, default=3)
    parser.add_argument("--maps", type=int, default=2, help="maps per opponent; both seats always played")
    parser.add_argument("--sizes", type=int, nargs="+", default=[12, 16, 20, 24])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export", type=Path, help="write a self-contained dashboard after training")
    args = parser.parse_args()
    if args.generations < 1 or args.population < 4 or not 1 <= args.opponents <= args.population or args.maps < 1 or args.workers < 1:
        parser.error("positive counts required; population >= 4 and opponents <= population")
    if any(s < 6 or s > 100 for s in args.sizes):
        parser.error("map sizes must be 6..100")
    subprocess.run(["cargo", "build", "--release", "--manifest-path", str(HERE / "Cargo.toml")], check=True)
    train(args)


if __name__ == "__main__":
    main()
