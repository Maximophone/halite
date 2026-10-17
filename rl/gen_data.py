#!/usr/bin/env python3
"""
Generate teacher games for behaviour cloning: replays where DexBot holds most seats.

Each game has 2 to 6 players. Every seat is DexBot with probability --teacher-share, otherwise
one of the other strong bots, and at least one seat is DexBot. The trainer learns only from the
teacher's seats (by name), so the other bots just make the positions it sees more varied.

    python3 rl/gen_data.py --games 400 --seed 1            # replays go to rl/data/replays/
    python3 rl/gen_data.py --games 50 --seed 2 --out rl/data/val

Map sizes are the engine's default (random square 20..50). Needs the Rust engine and bots built.
"""

import argparse
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "engine", "target", "release", "halite")
BIN = os.path.join(ROOT, "starter-kit", "rust", "target", "release")
TEACHER = os.path.join(BIN, "dex_bot")
OTHERS = [os.path.join(BIN, b) for b in ("apex_gamma", "hal_lite", "apex_beta", "predator_bot")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "rl", "data", "replays"))
    ap.add_argument("--teacher-share", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    for b in [ENGINE, TEACHER] + OTHERS:
        if not os.path.exists(b):
            sys.exit("missing %s: build the engine and starter-kit/rust first" % b)
    os.makedirs(a.out, exist_ok=True)

    rng = random.Random(a.seed)
    jobs = []
    for _ in range(a.games):
        n = rng.choice([2, 2, 3, 4, 4, 5, 6])
        seats = [TEACHER if rng.random() < a.teacher_share else rng.choice(OTHERS) for _ in range(n)]
        if TEACHER not in seats:
            seats[rng.randrange(n)] = TEACHER
        jobs.append((rng.randrange(2 ** 31), seats))

    def run(job):
        seed, seats = job
        args = [ENGINE, "-q", "-s", str(seed), "-o", a.out, "--no-html", "--no-logs", "--"] + seats
        return subprocess.run(args, capture_output=True, text=True).returncode

    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for rc in ex.map(run, jobs):
            done += 1
            if rc != 0:
                print("engine exited with", rc, file=sys.stderr)
            if done % 50 == 0:
                print("%d/%d games" % (done, a.games), flush=True)


if __name__ == "__main__":
    main()
