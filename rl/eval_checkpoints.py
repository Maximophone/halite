#!/usr/bin/env python3
"""
Evaluate RL checkpoints against the hand-written bots as they appear (standard library only).

    python3 rl/eval_checkpoints.py rl/runs/ppo1 --every 25
    python3 rl/eval_checkpoints.py rl/runs/solo1 --every 10 --solo --start rl/runs/bc4/best.hnn

Watches RUN for iterNNNN.hnn files (written by `halite-rl rl`), and for every iteration that is a
multiple of --every (plus the starting point, the released NeuroAlpha weights) plays the
NeuroAlpha binary with those weights through tools/eval_bot.py: 1v1 and 4-player games against
the PredatorBot pool, 1v1 against DexBot. Appends one line per checkpoint and format to
RUN/evals.txt. Same seeds for every checkpoint, so the numbers are comparable.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "starter-kit", "rust", "target", "release")
START = os.path.join(ROOT, "starter-kit", "rust", "weights", "neuro_alpha.hnn")


def evaluate_solo(weights, label, out):
    """Solo benchmark (tools/solo_bench.py): production owned at turns 60 / 100 / 150, alone."""
    cand = "NEURO_WEIGHTS=%s %s" % (weights, os.path.join(BIN, "neuro_alpha"))
    res = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "solo_bench.py"), cand], capture_output=True, text=True)
    m = re.search(r"turn 60:\s+([\d.]+)\s+turn 100:\s+([\d.]+)\s+turn 150:\s+([\d.]+)", res.stdout)
    if m:
        row = "%s  solo           production turn 60 %s  turn 100 %s  turn 150 %s" % (label, m.group(1), m.group(2), m.group(3))
        print(row, flush=True)
        with open(out, "a") as f:
            f.write(row + "\n")


def evaluate(weights, label, out, n, seed, solo=False):
    if solo:
        return evaluate_solo(weights, label, out)
    cand = "NEURO_WEIGHTS=%s %s" % (weights, os.path.join(BIN, "neuro_alpha"))
    runs = [("pool", ["--suite", "1v1,ffa4"]), ("dex", ["--suite", "1v1", "--opp", os.path.join(BIN, "dex_bot")])]
    for name, extra in runs:
        cmd = [sys.executable, os.path.join(ROOT, "tools", "eval_bot.py"), cand, "--n", str(n), "--seed", str(seed), "--workers", "6"] + extra
        res = subprocess.run(cmd, capture_output=True, text=True)
        for line in res.stdout.splitlines():
            m = re.match(r"(\S+)\s+win\s+([\d.]+)%\s+avg rank ([\d.]+)\s+\((\d+) games, (\d+) errors\)", line)
            if m:
                fmt = m.group(1) + (" vs DexBot" if name == "dex" else " vs pool")
                row = "%s  %-14s win %5s%%  avg rank %s  (%s games, %s errors)" % (label, fmt, m.group(2), m.group(3), m.group(4), m.group(5))
                print(row, flush=True)
                with open(out, "a") as f:
                    f.write(row + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--n", type=int, default=40, help="games per format")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--solo", action="store_true", help="solo benchmark instead of games against bots")
    ap.add_argument("--start", default=START, help="weights of the starting point (iter0000)")
    a = ap.parse_args()
    out = os.path.join(a.run, "evals.txt")
    done = set()
    if os.path.exists(out):
        done = {line.split()[0] for line in open(out)}
    if "iter0000" not in done:
        evaluate(a.start, "iter0000", out, a.n, a.seed, a.solo)
        done.add("iter0000")
    while True:
        for path in sorted(glob.glob(os.path.join(a.run, "iter*.hnn"))):
            label = os.path.basename(path)[:-4]
            if label not in done and int(label[4:]) % a.every == 0:
                evaluate(path, label, out, a.n, a.seed, a.solo)
                done.add(label)
        time.sleep(60)


if __name__ == "__main__":
    main()
