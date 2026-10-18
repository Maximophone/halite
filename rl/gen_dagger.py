#!/usr/bin/env python3
"""
DAgger games: the network plays one seat through rl/target/release/dagger, which records DexBot's
answers to the positions the network reaches (see rl/src/bin/dagger.rs). The trainer reads the
resulting .dag files next to the original teacher replays.

    python3 rl/gen_dagger.py --weights rl/runs/bc4/best.hnn --games 1000 --seed 1 --out rl/data/dagger1

Each game has 2 to 6 players; the other seats are DexBot with probability --dex-share, otherwise
ApexGamma, HAL-lite, ApexBeta or PredatorBot. No replays are written (the .dag files hold the
positions). The network runs on the CPU inside the bot, so games are slower than teacher games.
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
DAGGER = os.path.join(ROOT, "rl", "target", "release", "dagger")
DEX = os.path.join(BIN, "dex_bot")
OTHERS = [os.path.join(BIN, b) for b in ("apex_gamma", "hal_lite", "apex_beta", "predator_bot")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "rl", "data", "dagger"))
    ap.add_argument("--dex-share", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.0, help="chance per turn of playing DexBot's moves instead")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for b in [ENGINE, DAGGER, DEX] + OTHERS:
        if not os.path.exists(b):
            sys.exit("missing %s: build the engine, starter-kit/rust and rl first" % b)
    os.makedirs(a.out, exist_ok=True)
    weights = os.path.abspath(a.weights)
    out = os.path.abspath(a.out)
    learner = "DAGGER_OUT=%s NEURO_WEIGHTS=%s DAGGER_EXPERT=%s DAGGER_BETA=%s %s" % (out, weights, DEX, a.beta, DAGGER)

    rng = random.Random(a.seed)
    jobs = []
    for _ in range(a.games):
        n = rng.choice([2, 2, 2, 3, 4, 4, 5, 6])
        seats = [DEX if rng.random() < a.dex_share else rng.choice(OTHERS) for _ in range(n)]
        seats[rng.randrange(n)] = learner
        jobs.append((rng.randrange(2 ** 31), seats))

    def run(job):
        seed, seats = job
        args = [ENGINE, "-q", "-s", str(seed), "--no-replay", "--no-logs", "--"] + seats
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
