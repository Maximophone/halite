#!/usr/bin/env python3
"""
Where did each player's strength go? Replays a .hlt turn by turn and splits
every player's losses into: damage from each opponent (effective, capped at
what the piece had, shared between attackers in proportion to their damage),
strength spent on neutral sites, and strength wasted past the 255 cap.

Sites never change owner directly in practice: a killed piece leaves a neutral
empty site, and the front becomes a no-man's land that pieces step into.

    python3 tools/combat_report.py replays/x.hlt
    python3 tools/combat_report.py replays/x.hlt --every 25

Rows are windows of turns; "dealt" is the effective damage the player did to
all opponents, "lost" what opponents did to it. A bot that loses combat with
similar strength trades badly (micro); one that is simply outgrown loses on
economy (see tools/trace_game.py).
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replay")
    ap.add_argument("--every", type=int, default=25)
    a = ap.parse_args()
    with open(a.replay) as f:
        rp = json.load(f)
    W, H, P = rp["width"], rp["height"], rp["num_players"]
    n = W * H
    prod = rp["productions"]
    names = [s[:10] for s in rp["player_names"]]

    def nb(i, d):
        x, y = i % W, i // W
        if d == 1:
            return ((y - 1) % H) * W + x
        if d == 2:
            return y * W + (x + 1) % W
        if d == 3:
            return ((y + 1) % H) * W + x
        if d == 4:
            return y * W + (x - 1) % W
        return i

    def blank():
        return {"dmg": [[0.0] * (P + 1) for _ in range(P + 1)], "neutral": [0] * (P + 1),
                "cap": [0] * (P + 1)}

    win = blank()
    tot = blank()
    start = 0
    print("turns      " + " | ".join("%-10s dealt  lost neut  cap" % nm for nm in names))

    def flush(lo, hi, w):
        cols = []
        for p in range(1, P + 1):
            dealt = sum(w["dmg"][q][p] for q in range(1, P + 1))
            lost = sum(w["dmg"][p][q] for q in range(1, P + 1))
            cols.append("%-10s %5d %5d %4d %4d" % ("", dealt, lost, w["neutral"][p], w["cap"][p]))
        print("%4d-%-4d  " % (lo, hi) + " | ".join(cols))

    nf = len(rp["frames"])
    for f in range(nf - 1):
        owner, strength = rp["frames"][f]
        mv = rp["moves"][f]
        pieces = [dict() for _ in range(P + 1)]
        for i in range(n):
            o = owner[i]
            if o == 0:
                continue
            d = int(mv[i])
            if d == 0:
                s, dest = strength[i] + prod[i], i
            else:
                s, dest = strength[i], nb(i, d)
                pieces[o].setdefault(i, 0)
            tot_s = pieces[o].get(dest, 0) + s
            if tot_s > 255:
                win["cap"][o] += tot_s - 255
                tot_s = 255
            pieces[o][dest] = tot_s
        # damage[q][site] = list of (p, amount)
        incoming = [dict() for _ in range(P + 1)]
        for p in range(1, P + 1):
            for i, s in pieces[p].items():
                if s <= 0:
                    continue
                targets = [i] + [nb(i, d) for d in (1, 2, 3, 4)]
                for q in range(1, P + 1):
                    if q == p:
                        continue
                    for t in targets:
                        if t in pieces[q]:
                            incoming[q].setdefault(t, []).append((p, s))
                if owner[i] == 0 and strength[i] > 0:
                    win["neutral"][p] += min(s, strength[i])
        for q in range(1, P + 1):
            for t, lst in incoming[q].items():
                s = pieces[q][t]
                total = sum(x for _, x in lst)
                if total <= 0 or s <= 0:
                    continue
                eff = min(s, total)
                for p, x in lst:
                    win["dmg"][q][p] += eff * x / total
        if (f + 1) % a.every == 0 or f == nf - 2:
            flush(start, f + 1, win)
            for x in range(P + 1):
                for y in range(P + 1):
                    tot["dmg"][x][y] += win["dmg"][x][y]
            for k in ("neutral", "cap"):
                for x in range(P + 1):
                    tot[k][x] += win[k][x]
            win = blank()
            start = f + 1
    flush(0, nf - 1, tot)
    print("\neffective damage, row lost to column:")
    print("%-10s " % "" + " ".join("%10s" % nm for nm in names))
    for q in range(1, P + 1):
        print("%-10s " % names[q - 1] + " ".join("%10d" % tot["dmg"][q][p] for p in range(1, P + 1)))


if __name__ == "__main__":
    main()
