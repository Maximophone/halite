#!/usr/bin/env python3
"""
Play one game and print each player's territory, production and strength every
N turns: the fastest way to tell whether a loss came from economy or combat.
Writes the replay (.hlt and .html) so you can watch the turns that matter.

    python3 tools/trace_game.py 343436950 ./apex_bot starter-kit/rust/target/release/predator_bot
    python3 tools/trace_game.py 648096144 ./a ./b ./c ./d --every 40 -o replays

Player 1 is the first command (yellow in the viewer), player 2 blue, then red,
green, violet, orange. Debug output a bot writes to stderr goes to logs/.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from halite import play_game  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seed", type=int)
    ap.add_argument("bots", nargs="+")
    ap.add_argument("-d", "--dimensions", help='map size "W H" (default: from the seed, like the arena)')
    ap.add_argument("--every", type=int, default=20, help="print every N turns")
    ap.add_argument("-o", "--replay-dir", default=os.path.join(ROOT, "replays"))
    ap.add_argument("--engine", default="auto", choices=["auto", "python", "rust"])
    a = ap.parse_args()
    w = h = None
    if a.dimensions:
        w, h = (int(v) for v in a.dimensions.replace("x", " ").split())
    r = play_game(a.bots, w, h, a.seed, replay_dir=a.replay_dir, write_html=True,
                  log_dir=os.path.join(ROOT, "logs"), engine=a.engine)
    with open(r["replay"]) as f:
        rp = json.load(f)
    prod, names, n = rp["productions"], rp["player_names"], rp["num_players"]
    print("map %dx%d, %d turns" % (rp["width"], rp["height"], r["turns"]))
    frames = list(range(0, len(rp["frames"]), a.every))
    if frames[-1] != len(rp["frames"]) - 1:
        frames.append(len(rp["frames"]) - 1)
    for f in frames:
        owners, strengths = rp["frames"][f]
        cols = []
        for p in range(1, n + 1):
            sites = [i for i, o in enumerate(owners) if o == p]
            cols.append("%-10s t%5d p%5d s%6d" % (names[p - 1][:10], len(sites),
                                                  sum(prod[i] for i in sites), sum(strengths[i] for i in sites)))
        print("%4d  " % f + " | ".join(cols))
    for x in r["results"]:
        print("  #%d %s (%s)" % (x["rank"], x["name"], x["status"]))
    print("replay:", r["replay"].replace(".hlt", ".html"))


if __name__ == "__main__":
    main()
