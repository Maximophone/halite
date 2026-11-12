#!/usr/bin/env python3
"""
Evaluate a candidate against any lineup of opponents, many games in parallel.

    python3 tools/eval_lineup.py CAND --opps codex_bot                  # 1v1
    python3 tools/eval_lineup.py CAND --opps codex_bot,erdman_bot,dex_bot --n 200

CAND and each opponent are a bot command, or the stem of a binary in
starter-kit/rust/target/release. Seeds and seats come from --seed, so two
candidates are compared on the same maps and seats. Prints win rate, average
rank, errors of the candidate, and how often it finished ahead of each opponent.
"""

import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from halite import play_game  # noqa: E402

BIN = os.path.join(ROOT, "starter-kit", "rust", "target", "release")


def cmd(name):
    path = os.path.join(BIN, name)
    return path if os.path.exists(path) else name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cand")
    ap.add_argument("--opps", required=True, help="comma-separated opponents")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--size", help='fixed map size, e.g. "30 30"')
    ap.add_argument("--lost", action="store_true", help="print lost seeds")
    a = ap.parse_args()

    opps = a.opps.split(",")
    lineup = [cmd(a.cand)] + [cmd(o) for o in opps]
    rng = random.Random(a.seed * 1000 + len(lineup))
    jobs = []
    for _ in range(a.n):
        seed = rng.randrange(2 ** 31)
        order = list(range(len(lineup)))
        rng.shuffle(order)
        jobs.append((seed, order))

    def run(job):
        seed, order = job
        kw = {}
        if a.size:
            w, h = a.size.split()
            kw = {"width": int(w), "height": int(h)}
        r = play_game([lineup[i] for i in order], seed=seed, replay_dir=None, write_html=False,
                      log_dir=None, **kw)
        rank = {order[x["player"] - 1]: x["rank"] for x in r["results"]}
        err = any("(" in x["status"] for x in r["results"] if order[x["player"] - 1] == 0)
        return rank, err, seed, r["width"], r["height"]

    with ThreadPoolExecutor(a.workers) as ex:
        rs = list(ex.map(run, jobs))
    n = len(rs)
    wins = sum(r[0][0] == 1 for r in rs)
    se = 100 * (max(wins, 1) * max(n - wins, 1) / n ** 3) ** 0.5
    ahead = ["%s %.0f%%" % (opps[i - 1], 100.0 * sum(r[0][0] < r[0][i] for r in rs) / n)
             for i in range(1, len(lineup))]
    print("%dp win %5.1f%% (+-%.1f)  avg rank %.2f  %d games, %d errors  ahead of: %s" % (
        len(lineup), 100.0 * wins / n, se, sum(r[0][0] for r in rs) / n, n,
        sum(r[1] for r in rs), ", ".join(ahead)), flush=True)
    if a.lost:
        print("lost:", [(r[2], r[3], r[4], r[0][0]) for r in rs if r[0][0] != 1][:12])


if __name__ == "__main__":
    main()
