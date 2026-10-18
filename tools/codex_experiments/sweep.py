#!/usr/bin/env python3
"""Reproducible, paired-seat experiments, with complete per-game JSON results."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import random
import shlex
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from halite import play_game


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", help="JSON mapping experiment names to CODEX_* settings")
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--opponents", default="erdman_bot", help="comma-separated pool")
    ap.add_argument("--candidate", default="starter-kit/rust/target/codex-build/release/codex_bot")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.games < 1 or a.workers < 1 or not 2 <= a.players <= 6:
        ap.error("games and workers must be positive; players must be between 2 and 6")
    configs = json.loads(Path(a.configs).read_text())
    rng = random.Random(a.seed)
    cases = [(rng.randrange(2**31), rng.choice([20, 25, 30, 35, 40, 45, 50]))
             for _ in range((a.games + a.players - 1) // a.players)]
    pool = a.opponents.split(",")
    binaries = [ROOT / a.candidate] + [ROOT / "starter-kit/rust/target/release" / b for b in pool]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = (out.parent / "binaries").resolve()
    snapshot_dir.mkdir(exist_ok=True)
    frozen = {}
    hashes = {}
    # Other agents and the arena may rebuild binaries while this experiment runs.
    # Every match in one experiment uses the same immutable executable snapshot.
    for p in binaries:
        data = p.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        hashes[str(p)] = digest
        target = snapshot_dir / f"{p.name}-{digest[:16]}"
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=snapshot_dir, delete=False) as tmp:
                tmp.write(data)
                temporary = Path(tmp.name)
            shutil.copymode(p, temporary)
            temporary.replace(target)
        frozen[str(p)] = str(target)
    results = {"arguments": vars(a), "sha256": {
        str(p): hashes[str(p)] for p in binaries
    }, "snapshots": frozen, "experiments": {}}
    for name, settings in configs.items():
        command = " ".join([f"CODEX_{k}={shlex.quote(str(v))}" for k, v in settings.items()]
                           + [shlex.quote(frozen[str(ROOT / a.candidate)])])
        jobs = [(i, seed, side, seat) for i, (seed, side) in enumerate(cases)
                for seat in range(a.players)][:a.games]

        def run(job):
            i, seed, side, seat = job
            opponents = [pool[(i + j) % len(pool)] for j in range(a.players - 1)]
            cmds = [frozen[str(ROOT / "starter-kit/rust/target/release" / b)] for b in opponents]
            cmds.insert(seat, command)
            r = play_game(cmds, side, side, seed, replay_dir=None, write_html=False,
                          log_dir=None, engine="rust")
            mine = next(x for x in r["results"] if x["player"] == seat + 1)
            return {"seed": seed, "size": side, "width": r["width"], "height": r["height"],
                    "seat": seat + 1, "opponents": opponents,
                    "rank": mine["rank"], "status": mine["status"], "turns": r["turns"],
                    "results": r["results"]}

        started = time.monotonic()
        with ThreadPoolExecutor(a.workers) as ex:
            games = list(ex.map(run, jobs))
        wins = sum(g["rank"] == 1 for g in games)
        errors = sum("(" in g["status"] for g in games)
        opponent_errors = sum("(" in r["status"] for g in games for r in g["results"]
                              if r["player"] != g["seat"])
        result = {"settings": settings, "wins": wins, "games": games,
                  "errors": errors, "opponent_errors": opponent_errors,
                  "seconds": round(time.monotonic() - started, 2)}
        results["experiments"][name] = result
        out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"{name}: {wins}/{len(games)} wins ({100*wins/len(games):.1f}%), "
              f"rank {sum(g['rank'] for g in games)/len(games):.2f}, "
              f"{errors} candidate / {opponent_errors} opponent errors, {result['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
