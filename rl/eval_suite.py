#!/usr/bin/env python3
"""
The standard evaluation of a neural bot candidate, always on the same games (stdlib only).

    python3 rl/eval_suite.py "starter-kit/rust/target/release/neuro_beta" --label neurobeta-bc3
    python3 rl/eval_suite.py "NEURO_WEIGHTS=rl/runs/ei1/model.hnn starter-kit/rust/target/release/neuro_beta" --label ei1

Six formats, 100 games each, tools/eval_bot.py --seed 23: 1v1 / ffa4 / ffa6 against the PredatorBot
pool, then 1v1 against DexBot, ApexGamma and ErdmanBot. Prints the results and appends one line to
rl/checkpoints/suite.txt, so candidates line up side by side. One standard error at 100 games is
about 5 points near 50%, about 3 near 90%.
"""

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "starter-kit", "rust", "target", "release")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cand")
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    runs = [("pool", "1v1,ffa4,ffa6", None), ("dex", "1v1", "dex_bot"), ("gamma", "1v1", "apex_gamma"), ("erdman", "1v1", "erdman_bot")]
    cols = []
    for name, suite, opp in runs:
        cmd = [sys.executable, os.path.join(ROOT, "tools", "eval_bot.py"), a.cand, "--n", str(a.n), "--seed", str(a.seed),
               "--workers", str(a.workers), "--suite", suite]
        if opp:
            cmd += ["--opp", os.path.join(BIN, opp)]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        for line in out.splitlines():
            m = re.match(r"(\S+)\s+win\s+([\d.]+)%.*\((\d+) games, (\d+) errors\)", line)
            if m:
                label = m.group(1) if name == "pool" else "1v1 " + {"dex": "DexBot", "gamma": "ApexGamma", "erdman": "ErdmanBot"}[name]
                cols.append("%s %s%%%s" % (label, m.group(2), "" if m.group(4) == "0" else " (%s errors)" % m.group(4)))
                print("%-16s %s" % (label, line.strip()[:70]), flush=True)
    row = "%-20s  %s" % (a.label, "  |  ".join(cols))
    os.makedirs(os.path.join(ROOT, "rl", "checkpoints"), exist_ok=True)
    with open(os.path.join(ROOT, "rl", "checkpoints", "suite.txt"), "a") as f:
        f.write(row + "\n")
    print(row)


if __name__ == "__main__":
    main()
