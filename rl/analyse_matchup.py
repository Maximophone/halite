#!/usr/bin/env python3
"""
Where does a bot lose a 1v1 matchup: economy before contact, or the fights after it?

    python3 rl/analyse_matchup.py --games 20 --seed 31            # NeuroAlpha vs DexBot
    python3 rl/analyse_matchup.py --a "NEURO_WEIGHTS=w.hnn starter-kit/rust/target/release/neuro_alpha"

Plays 1v1 games with replays (seats alternate), then per game: the first turn the two players'
territories touch, each player's production at that turn and 60 turns later, the winner, and over
the whole game the strength each lost to the other against the strength it spent on neutral sites
(same accounting as tools/combat_report.py). Averages for won and lost games of player A.
"""

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "starter-kit", "rust", "target", "release")
ENGINE = os.path.join(ROOT, "engine", "target", "release", "halite")


def neighbours(i, w, h):
    x, y = i % w, i // w
    return [((y - 1) % h) * w + x, ((y + 1) % h) * w + x, y * w + (x + 1) % w, y * w + (x - 1) % w]


def analyse(path, a_seat):
    r = json.load(open(path))
    w, h = r["width"], r["height"]
    prod = r["productions"]
    frames = r["frames"]
    b_seat = 3 - a_seat
    contact = None
    for f, (own, _) in enumerate(frames):
        if any(own[i] == a_seat and any(own[j] == b_seat for j in neighbours(i, w, h)) for i in range(w * h)):
            contact = f
            break

    def production(f, p):
        f = min(f, len(frames) - 1)
        return sum(prod[i] for i, o in enumerate(frames[f][0]) if o == p)

    # strength lost to the opponent vs spent on neutrals, turn by turn (see tools/combat_report.py)
    lost = {1: 0, 2: 0}
    neutral = {1: 0, 2: 0}
    for f in range(len(frames) - 1):
        own, st = frames[f]
        nown, nst = frames[f + 1]
        mv = r["moves"][f]
        for p in (1, 2):
            q = 3 - p
            for i in range(w * h):
                if own[i] != p:
                    continue
                d = int(mv[i])
                dest = i if d == 0 else neighbours(i, w, h)[[None, 0, 2, 1, 3][d]]
                if nown[dest] == 0 and own[dest] == 0 and d != 0:
                    neutral[p] += min(st[i], st[dest])
                if nown[dest] == q:
                    lost[p] += st[i]
    a_won = next(x for x in r["results"] if x["player"] == a_seat)["rank"] == 1
    return {
        "won": a_won, "turns": len(frames) - 1, "contact": contact,
        "prod_contact": (production(contact or 0, a_seat), production(contact or 0, b_seat)),
        "prod_after": (production((contact or 0) + 60, a_seat), production((contact or 0) + 60, b_seat)),
        "lost": (lost[a_seat], lost[b_seat]), "neutral": (neutral[a_seat], neutral[b_seat]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default=os.path.join(BIN, "neuro_alpha"))
    ap.add_argument("--b", default=os.path.join(BIN, "dex_bot"))
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args()
    out = tempfile.mkdtemp(prefix="matchup-")
    rng = random.Random(a.seed)
    jobs = [(rng.randrange(2 ** 31), k % 2 == 0) for k in range(a.games)]

    def play(job):
        seed, a_first = job
        d = os.path.join(out, str(seed))
        bots = [a.a, a.b] if a_first else [a.b, a.a]
        subprocess.run([ENGINE, "-q", "-s", str(seed), "-o", d, "--no-html", "--no-logs", "--"] + bots, capture_output=True)
        return analyse(glob.glob(os.path.join(d, "*.hlt"))[0], 1 if a_first else 2)

    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(play, jobs))
    for label, sel in (("A won", [x for x in rows if x["won"]]), ("A lost", [x for x in rows if not x["won"]])):
        if not sel:
            continue
        m = lambda f: sum(f(x) for x in sel) / len(sel)
        print("%-7s %2d games  contact turn %5.1f  production at contact A %6.1f / B %6.1f   60 turns later A %6.1f / B %6.1f"
              % (label, len(sel), m(lambda x: x["contact"] or 0), m(lambda x: x["prod_contact"][0]), m(lambda x: x["prod_contact"][1]),
                 m(lambda x: x["prod_after"][0]), m(lambda x: x["prod_after"][1])))
        print("        strength lost to the opponent A %7.0f / B %7.0f   spent on neutrals A %7.0f / B %7.0f   game length %5.1f"
              % (m(lambda x: x["lost"][0]), m(lambda x: x["lost"][1]), m(lambda x: x["neutral"][0]), m(lambda x: x["neutral"][1]), m(lambda x: x["turns"])))


if __name__ == "__main__":
    main()
