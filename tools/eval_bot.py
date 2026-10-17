#!/usr/bin/env python3
"""
Evaluate a candidate bot against a fixed opponent pool, many games in parallel.

Seeds and seat orders come from --seed, so two versions of a bot are compared
on exactly the same maps. Reports win rate, average rank and errors (crashes
or timeouts: a crashing bot looks like a weak bot) per format, plus a few
lost seeds to trace with tools/trace_game.py.

    python3 tools/eval_bot.py starter-kit/rust/target/release/apex_alpha
    python3 tools/eval_bot.py "APEX_DANGER=7 ./bots/apex_v12" --n 100 --suite 1v1,ffa4
    python3 tools/eval_bot.py ./candidate --opp starter-kit/rust/target/release/apex_alpha

The candidate command may set environment variables, since bots run through a
shell. Uses the Rust engine when it is built. Noise: at 60 games a win rate
moves by about 6 points from luck alone; use 100+ for decisions.
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
PRED, IMP, RND = (os.path.join(BIN, b) for b in ("predator_bot", "improved_bot", "random_bot"))


def lineups(cand, opp):
    return {
        "1v1": [cand, opp],
        "ffa3": [cand, opp, IMP],
        "ffa4": [cand, opp, opp, IMP],
        "ffa6": [cand, opp, opp, IMP, IMP, RND],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cand", help="candidate bot command")
    ap.add_argument("--n", type=int, default=60, help="games per format (default 60)")
    ap.add_argument("--suite", default="1v1,ffa4,ffa6", help="formats: 1v1, ffa3, ffa4, ffa6")
    ap.add_argument("--seed", type=int, default=1, help="seed set; change it to check on fresh maps")
    ap.add_argument("--opp", default=PRED, help="main opponent (default PredatorBot)")
    ap.add_argument("--workers", type=int, default=8, help="games in parallel")
    ap.add_argument("--engine", default="auto", choices=["auto", "python", "rust"])
    a = ap.parse_args()

    for name in a.suite.split(","):
        lineup = lineups(a.cand, a.opp)[name]
        rng = random.Random(a.seed * 1000 + len(lineup))
        jobs = []
        for _ in range(a.n):
            seed = rng.randrange(2 ** 31)
            order = list(range(len(lineup)))
            rng.shuffle(order)  # the candidate's seat varies
            jobs.append((seed, [lineup[i] for i in order], order.index(0) + 1))

        def run(job):
            seed, cmds, me = job
            r = play_game(cmds, seed=seed, replay_dir=None, write_html=False, log_dir=None, engine=a.engine)
            mine = next(x for x in r["results"] if x["player"] == me)
            return mine["rank"], r["width"], seed, "(" in mine["status"]

        with ThreadPoolExecutor(a.workers) as ex:
            rs = list(ex.map(run, jobs))
        wins = sum(r[0] == 1 for r in rs)
        losses = [(seed, rank, size) for rank, size, seed, _ in rs if rank != 1]
        print("%-5s win %5.1f%%  avg rank %.2f  (%d games, %d errors)  lost (seed, rank, size): %s" % (
            name, 100.0 * wins / len(rs), sum(r[0] for r in rs) / len(rs), len(rs),
            sum(r[3] for r in rs), losses[:5]), flush=True)


if __name__ == "__main__":
    main()
