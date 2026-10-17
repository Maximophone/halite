#!/usr/bin/env python3
"""
Run many games between bots on random maps and report win rates and
average finishing positions.

Usage:
    python3 tournament.py -n 20 "python3 MyBot.py" "python3 ImprovedBot.py"
    python3 tournament.py -n 10 --players 4 "python3 A.py" "python3 B.py" "python3 C.py" "python3 D.py"

With --players smaller than the number of bots, each game samples that many
bots at random. Replays are only kept with --keep-replays.
"""

import argparse
import random
from collections import defaultdict

from halite import play_game, resolve_engine


def main():
    ap = argparse.ArgumentParser(description="Halite I clone tournament runner")
    ap.add_argument("bots", nargs="+")
    ap.add_argument("-n", "--games", type=int, default=10)
    ap.add_argument("--players", type=int, help="bots per game (default: all)")
    ap.add_argument("-d", "--dimensions", help='fixed map size "W H" (default random)')
    ap.add_argument("--keep-replays", action="store_true")
    ap.add_argument("-s", "--seed", type=int, help="seed for the sequence of games")
    ap.add_argument("--engine", choices=["auto", "python", "rust"], default="auto",
                    help="game engine (default: Rust when engine/ is built, else Python)")
    args = ap.parse_args()
    engine = resolve_engine(args.engine)
    print("Engine: %s" % engine)

    rng = random.Random(args.seed)
    per_game = args.players or len(args.bots)
    w = h = None
    if args.dimensions:
        w, h = (int(v) for v in args.dimensions.replace("x", " ").split())

    games = defaultdict(int)
    wins = defaultdict(int)
    rank_sum = defaultdict(int)
    names = {}

    for g in range(1, args.games + 1):
        lineup = rng.sample(args.bots, per_game)
        res = play_game(lineup, w, h, rng.randrange(2 ** 31),
                        replay_dir="replays" if args.keep_replays else None,
                        write_html=args.keep_replays, engine=engine)
        line = []
        for r in res["results"]:
            cmd = r["command"]
            names[cmd] = r["name"]
            games[cmd] += 1
            rank_sum[cmd] += r["rank"]
            if r["rank"] == 1:
                wins[cmd] += 1
            line.append("%d. %s" % (r["rank"], r["name"]))
        print("Game %3d  %dx%d  %3d turns  %s" % (g, res["width"], res["height"], res["turns"], "  ".join(line)), flush=True)

    print("\n%-28s %6s %6s %8s %9s" % ("bot", "games", "wins", "win %", "avg rank"))
    for cmd in sorted(games, key=lambda c: rank_sum[c] / games[c]):
        print("%-28s %6d %6d %7.0f%% %9.2f" % (
            names[cmd][:28], games[cmd], wins[cmd], 100.0 * wins[cmd] / games[cmd], rank_sum[cmd] / games[cmd]))


if __name__ == "__main__":
    main()
