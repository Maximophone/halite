#!/usr/bin/env python3
"""Solo economy of several bot commands on the same N maps (seed 99), relative to the first:
mean production at turns 60/100/150 and the number of maps where a bot has under half of the
first bot's production at turn 100 (a stalled start).

    python3 tools/solo_compare.py --maps 160 CMD_A CMD_B ...
"""
import argparse, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solo_bench as sb
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("bots", nargs="+")
ap.add_argument("--maps", type=int, default=160)
ap.add_argument("--seed", type=int, default=99)
a = ap.parse_args()
rng = random.Random(a.seed)
cases = [(rng.randrange(2 ** 31), rng.choice([20, 25, 30, 35, 40, 45, 50])) for _ in range(a.maps)]
base = None
for b in a.bots:
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda c: sb.run(b, *c), cases))
    if base is None:
        base = res
    means = [sum(r[k] for r in res) / len(res) for k in range(3)]
    stalled = sum(r[1] < 0.5 * q[1] for r, q in zip(res, base))
    print("%-70s %7.1f %7.1f %7.1f  stalled %d" % (b[-70:], *means, stalled), flush=True)
