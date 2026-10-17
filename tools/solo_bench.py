#!/usr/bin/env python3
"""
Expansion speed, with no opponent: each bot plays alone on the same fixed maps
and we report the mean production it owns at turns 60, 100 and 150.

No opponent and no randomness, so a 2% difference is real. This is the best
proxy for economy changes; it found the gathering planner and its ranking
exponent. Baselines (40 maps): ImprovedBot 672 / 1643 at turns 100 / 150,
PredatorBot 818 / 2116, ApexAlpha about 980 / 2565.

    python3 tools/solo_bench.py starter-kit/rust/target/release/apex_alpha
    python3 tools/solo_bench.py "APEX_GATHER=4 ./apex_bot" "APEX_GATHER=8 ./apex_bot"

Needs the Rust engine (cd engine && cargo build --release).
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "engine", "target", "release", "halite")
TURNS = (60, 100, 150)


def run(cmd, seed, side):
    d = tempfile.mkdtemp(prefix="solo-")
    try:
        subprocess.run([ENGINE, "-q", "-s", str(seed), "-d", "%d %d" % (side, side), "--turns", str(TURNS[-1]),
                        "-o", d, "--no-html", "--no-logs", cmd], capture_output=True, check=True)
        with open(os.path.join(d, os.listdir(d)[0])) as f:
            rp = json.load(f)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    prod = rp["productions"]
    return [sum(prod[i] for i, o in enumerate(rp["frames"][t][0]) if o == 1) for t in TURNS]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bots", nargs="+", help="bot commands")
    ap.add_argument("--maps", type=int, default=40, help="number of maps (default 40)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if not os.access(ENGINE, os.X_OK):
        sys.exit("Rust engine not built: cd engine && cargo build --release")
    rng = random.Random(7)
    cases = [(rng.randrange(2 ** 31), rng.choice([20, 25, 30, 35, 40, 45, 50])) for _ in range(a.maps)]
    for cmd in a.bots:
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(lambda c: run(cmd, *c), cases))
        means = [sum(r[k] for r in res) / len(res) for k in range(len(TURNS))]
        print("%s\n    " % cmd + "   ".join("turn %d: %7.1f" % (t, m) for t, m in zip(TURNS, means)), flush=True)


if __name__ == "__main__":
    main()
