#!/usr/bin/env python3
"""
Check that the Rust ErdmanBot plays exactly like Travis Erdman's original erdman_v26.py.

Each case is played twice on the same map and seats: once with the original Python bot (after
random.seed(S), injected without editing his file) and once with the Rust port (ERDMAN_SEED=S).
The replays must be identical frame for frame. Opponents must be deterministic (not random_bot,
which seeds itself from the clock). Turn timeouts are off, since the Python bot is slower.

His repository has no license, so it is not vendored; clone it first:

    git clone https://github.com/erdman/erdman-halite-bots /tmp/erdman
    python3 tools/check_erdman.py /tmp/erdman/erdman_v26.py
    python3 tools/check_erdman.py /tmp/erdman/erdman_v26.py --case 5:50:dex_bot,apex_gamma,hal_lite
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "starter-kit", "rust", "target", "release")
ENGINE = os.path.join(ROOT, "engine", "target", "release", "halite")

# Runs the original unchanged, after seeding Python's global random generator.
RUNNER = """
import os, random, runpy, sys
path, seed = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
random.seed(seed)
sys.argv = [path]
runpy.run_path(path, run_name="__main__")
"""

DEFAULT_CASES = [
    "1:20:predator_bot", "2:30:dex_bot", "3:40:apex_gamma", "4:25:hal_lite,predator_bot",
    "5:50:dex_bot,apex_gamma,hal_lite", "6:35:predator_bot,improved_bot,improved_bot,dex_bot,apex_gamma",
    "7:45:dex_bot", "8:24:apex_gamma,dex_bot,hal_lite", "9:50:apex_gamma", "10:20:dex_bot,dex_bot,dex_bot",
    "11:42:hal_lite,hal_lite,apex_gamma,dex_bot,predator_bot",
]


def play(cmds, map_seed, size, work):
    out = subprocess.run(
        [ENGINE, "-q", "-t", "--no-html", "-o", work, "-d", f"{size} {size}", "-s", str(map_seed),
         "--log-dir", os.path.join(work, "logs")] + cmds,
        capture_output=True, text=True, cwd=work, check=True)  # cwd: the bot writes roibot.log
    res = json.loads(out.stdout)
    path = res["replay"] if os.path.isabs(res["replay"]) else os.path.join(work, res["replay"])
    with open(path) as f:
        rep = json.load(f)
    os.remove(path)
    return res, rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original", help="path to erdman_v26.py (hlt.py must sit next to it)")
    ap.add_argument("--case", action="append", help="mapseed:size:opp1,opp2 (repeatable; default: 11 cases)")
    a = ap.parse_args()
    if not os.path.exists(ENGINE):
        sys.exit("build the Rust engine first: cd engine && cargo build --release")
    original = os.path.abspath(a.original)
    port = os.path.join(BIN, "erdman_bot")
    bad = 0
    with tempfile.TemporaryDirectory() as work:
        runner = os.path.join(work, "seeded_runner.py")
        with open(runner, "w") as f:
            f.write(RUNNER)
        for case in a.case or DEFAULT_CASES:
            map_seed, size, opps = case.split(":")
            rseed = int(map_seed) * 7919 + 13
            others = [os.path.join(BIN, o) for o in opps.split(",")]
            ra, orig = play([f"python3 {runner} {original} {rseed}"] + others, map_seed, size, work)
            rb, rust = play([f"ERDMAN_SEED={rseed} {port}"] + others, map_seed, size, work)
            same = orig["frames"] == rust["frames"] and orig["moves"] == rust["moves"]
            first = next((i for i, (x, y) in enumerate(zip(orig["moves"], rust["moves"])) if x != y), None)
            bad += not same
            print(f"{case}: {ra['turns']} / {rb['turns']} turns, identical={same}"
                  + ("" if same else f", first different move at turn {first}"), flush=True)
    print("all identical" if not bad else f"{bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
