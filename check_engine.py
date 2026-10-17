#!/usr/bin/env python3
"""
Prove that the Rust engine (engine/target/release/halite) plays exactly the
same games as the reference Python engine (halite.py).

Four checks:
  1. maps:  generate_map for many seeds, sizes and player counts (including
            the default random size) must give identical maps and starts;
  2. rules: GameState.step against the Rust step on thousands of random
            states plus the edge cases from test_rules.py;
  3. games: full games with deterministic bots on both engines; replays
            (every frame, every move, results), the -q summary and the console
            output must be identical;
  4. ejection: bots that crash, time out, close their pipe, die during init or
            send garbage must be handled identically.

Usage:
    python3 check_engine.py            # everything (a few minutes)
    python3 check_engine.py --quick    # fewer cases
    python3 check_engine.py --only rules

Build first: (cd engine && cargo build --release) and
(cd starter-kit/rust && cargo build --release) for the Rust PredatorBot.
"""

import argparse
import concurrent.futures as cf
import glob
import json
import os
import random
import re
import subprocess
import sys
import tempfile

import halite
from halite import GameState, generate_map

HERE = os.path.dirname(os.path.abspath(__file__))
RUST_ENGINE = os.path.join(HERE, "engine", "target", "release", "halite")
PY_ENGINE = [sys.executable, os.path.join(HERE, "halite.py")]

RUST_PREDATOR = "starter-kit/rust/target/release/predator_bot"
RUST_IMPROVED = "starter-kit/rust/target/release/improved_bot"


# ---------------------------------------------------------------------------
# 1. Maps
# ---------------------------------------------------------------------------

def python_map(w, h, n, seed):
    """The start of run_game: seeded rng, default size, generate_map."""
    rng = random.Random(seed)
    if w is None or h is None:
        side = rng.choice([20, 25, 30, 35, 40, 45, 50])
        w = w or side
        h = h or side
    W, H, prod, strength, starts = generate_map(w, h, n, rng)
    out = [W, H] + [v for row in prod for v in row] + [v for row in strength for v in row]
    for x, y in starts:
        out += [x, y]
    return " ".join(map(str, out))


def check_maps(quick):
    rng = random.Random(1)
    cases = []
    sizes = [None, (20, 20), (25, 25), (30, 30), (40, 40), (50, 50), (25, 40), (40, 25),
             (13, 7), (3, 3), (1, 1), (0, 30), (-5, 12), (64, 33), (100, 100)]
    seeds = [0, 1, 2, 7, 42, 2 ** 31 - 1, 2 ** 32, 2 ** 40 + 3, 2 ** 70 + 11, -1, -123456789]
    seeds += [rng.randrange(2 ** 31) for _ in range(10 if quick else 60)]
    for seed in seeds:
        for n in range(1, 7):
            for size in sizes:
                if size is None:
                    cases.append((None, None, n, seed))
                elif rng.random() < (0.3 if quick else 1.0):
                    cases.append((size[0], size[1], n, seed))
    lines = ["%s %s %d %d" % ("-" if w is None else w, "-" if h is None else h, n, s)
             for w, h, n, s in cases]
    out = subprocess.run([RUST_ENGINE, "--map-test"], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(out) == len(cases), (len(out), len(cases))
    for case, got in zip(cases, out):
        want = python_map(*case)
        if got != want:
            fail("map differs for (w, h, players, seed) = %r" % (case,))
    print("maps: %d maps identical" % len(cases))


# ---------------------------------------------------------------------------
# 2. Rules
# ---------------------------------------------------------------------------

def state_from(w, h, n, owner, strength, production):
    g = GameState(w, h, [production[y * w:(y + 1) * w] for y in range(h)],
                  [strength[y * w:(y + 1) * w] for y in range(h)], [])
    g.num_players = n
    g.owner = [owner[y * w:(y + 1) * w] for y in range(h)]
    return g


def python_step(w, h, n, owner, strength, production, dirs):
    g = state_from(w, h, n, owner, strength, production)
    moves = {}
    for i, d in enumerate(dirs):
        if owner[i]:
            moves.setdefault(owner[i], {})[(i % w, i // w)] = d
    g.step(moves)
    return " ".join(map(str, halite._flat(g.owner) + halite._flat(g.strength)))


def random_case(rng):
    w, h = rng.choice([1, 2, 3, 3, 4, 5, 6, 8, 10]), rng.choice([1, 2, 3, 3, 4, 5, 6, 8, 10])
    n = rng.randint(1, 6)
    size = w * h
    density = rng.random()

    def strength_value():
        r = rng.random()
        if r < 0.25:
            return 0
        if r < 0.35:
            return 255
        if r < 0.5:
            return rng.randint(250, 255)
        if r < 0.7:
            return rng.randint(1, 10)
        return rng.randint(0, 255)

    owner = [rng.randint(1, n) if rng.random() < density else 0 for _ in range(size)]
    strength = [strength_value() for _ in range(size)]
    production = [rng.choice([0, 0, 1, 5, 15, rng.randint(0, 20)]) for _ in range(size)]
    still_bias = rng.random()
    dirs = [0 if rng.random() < still_bias else rng.randint(0, 4) for _ in range(size)]
    return w, h, n, owner, strength, production, dirs


def rule_edge_cases():
    """The situations of test_rules.py, plus a few more, as flat states."""
    def make(w, h, cells, prod=0, moves=None, n=None):
        owner, strength = [0] * (w * h), [0] * (w * h)
        for (x, y), (o, s) in cells.items():
            owner[y * w + x], strength[y * w + x] = o, s
        dirs = [0] * (w * h)
        for (x, y), d in (moves or {}).items():
            dirs[y * w + x] = d
        n = n or max([o for o, _ in cells.values()] + [1])
        return w, h, n, owner, strength, [prod] * (w * h), dirs

    N, E, S, W_ = halite.NORTH, halite.EAST, halite.SOUTH, halite.WEST
    return [
        make(5, 5, {(1, 1): (1, 250)}, prod=10),                                  # cap on STILL
        make(5, 5, {(1, 1): (1, 50), (2, 1): (0, 20)}, moves={(1, 1): E}),        # capture
        make(5, 5, {(1, 1): (1, 10), (2, 1): (0, 30)}, moves={(1, 1): E}),        # failed capture
        make(5, 5, {(1, 1): (1, 30), (2, 1): (0, 30)}, moves={(1, 1): E}),        # both die
        make(5, 5, {(0, 1): (1, 200), (2, 1): (1, 200), (1, 1): (1, 0)},
             moves={(0, 1): E, (2, 1): W_}),                                      # merge cap 255
        make(5, 5, {(2, 2): (1, 100), (2, 1): (2, 40), (3, 2): (2, 40), (2, 3): (2, 40)}),  # overkill
        make(5, 5, {(1, 1): (1, 5), (2, 1): (0, 200)}),                           # no neutral adjacency
        make(4, 4, {(0, 0): (1, 50)}, moves={(0, 0): W_}),                        # wrap west
        make(4, 4, {(3, 0): (1, 50)}, moves={(3, 0): N}),                         # wrap north
        # Zero-strength pieces of two owners meeting on a neutral 0 site.
        make(5, 5, {(1, 1): (1, 0), (3, 1): (2, 0), (2, 1): (0, 0)}, moves={(1, 1): E, (3, 1): W_}),
        # Zero-strength piece moving onto a strong neutral: neutral holds.
        make(5, 5, {(1, 1): (1, 0), (2, 1): (0, 40)}, moves={(1, 1): E}),
        # Zero piece left behind next to an enemy: survives if undamaged only.
        make(5, 5, {(1, 1): (1, 60), (3, 1): (2, 5)}, moves={(1, 1): E}),
        # Six players around one site.
        make(5, 5, {(2, 2): (1, 255), (2, 1): (2, 255), (3, 2): (3, 255), (2, 3): (4, 255),
                    (1, 2): (5, 255), (0, 0): (6, 1)}, n=6),
        # Tiny maps where neighbours coincide (damage counted per target slot).
        make(1, 1, {(0, 0): (1, 9)}, moves={(0, 0): N}),
        make(2, 1, {(0, 0): (1, 9), (1, 0): (2, 4)}),
        make(2, 2, {(0, 0): (1, 100), (1, 1): (2, 100)}, moves={(0, 0): E, (1, 1): W_}),
        make(3, 3, {(0, 0): (1, 255), (1, 0): (1, 255), (2, 0): (1, 255)},
             moves={(0, 0): E, (2, 0): W_}),
    ]


def check_rules(quick):
    rng = random.Random(2)
    cases = rule_edge_cases() + [random_case(rng) for _ in range(2000 if quick else 20000)]
    lines = []
    for w, h, n, owner, strength, production, dirs in cases:
        lines.append(" ".join(map(str, [w, h, n] + owner + strength + production + dirs)))
    out = subprocess.run([RUST_ENGINE, "--step-test"], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(out) == len(cases), (len(out), len(cases))
    for case, got in zip(cases, out):
        want = python_step(*case)
        if got != want:
            fail("step differs for state %r\npython: %s\nrust:   %s" % (case, want, got))
    print("rules: %d steps identical (%d edge cases)" % (len(cases), len(rule_edge_cases())))


# ---------------------------------------------------------------------------
# 3 and 4. Games
# ---------------------------------------------------------------------------

def run_engine(engine, bots, args, workdir):
    """Run one game in `workdir` (so logs/ stays out of the way)."""
    replay_dir = os.path.join(workdir, "replays")
    cmd = (PY_ENGINE if engine == "python" else [RUST_ENGINE]) + args + ["-o", replay_dir] + bots
    os.makedirs(workdir, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir)
    hlt = glob.glob(os.path.join(replay_dir, "*.hlt"))
    html = glob.glob(os.path.join(replay_dir, "*.html"))
    replay = open(hlt[0]).read() if hlt else None
    page = open(html[0]).read() if html else None
    # Replay names carry the local time; the rest of the output must match.
    stdout = re.sub(r"\d{8}-\d{6}-", "STAMP-", proc.stdout.replace(replay_dir, "DIR"))
    return {"code": proc.returncode, "stdout": stdout, "replay": replay, "html": page}


def bot_cmd(cmd):
    """Bots run from each game's own directory: make relative paths absolute."""
    parts = cmd.split(" ")
    return " ".join(os.path.join(HERE, p) if p.startswith("starter-kit/") else p for p in parts)


def compare_game(label, bots, args, tmp):
    bots = [bot_cmd(b) for b in bots]
    py = run_engine("python", bots, args, os.path.join(tmp, label, "py"))
    rs = run_engine("rust", bots, args, os.path.join(tmp, label, "rs"))
    problems = []
    if py["code"] != rs["code"]:
        problems.append("exit code %s vs %s" % (py["code"], rs["code"]))
    if py["stdout"] != rs["stdout"]:
        problems.append("console output differs:\n--- python\n%s--- rust\n%s" % (py["stdout"], rs["stdout"]))
    if py["replay"] != rs["replay"]:
        problems.append(describe_replay_diff(py["replay"], rs["replay"]))
    if py["html"] != rs["html"]:
        problems.append("embedded HTML differs")
    turns = None
    if py["replay"]:
        turns = json.loads(py["replay"])["num_frames"] - 1
    elif "-q" in args and py["stdout"].strip():
        turns = json.loads(py["stdout"])["turns"]
    return label, problems, turns


def describe_replay_diff(a, b):
    if a is None or b is None:
        return "replay missing (python %s, rust %s)" % (a is not None, b is not None)
    ja, jb = json.loads(a), json.loads(b)
    if ja == jb:
        return "replay JSON text differs (same data, different formatting)"
    for key in ja:
        if ja[key] != jb.get(key):
            if key == "frames":
                for f, (fa, fb) in enumerate(zip(ja["frames"], jb["frames"])):
                    if fa != fb:
                        return "replays first differ at frame %d" % f
            return "replays differ in %r: %r vs %r" % (key, str(ja[key])[:200], str(jb.get(key))[:200])
    return "replays differ"


def game_cases(quick):
    rng = random.Random(3)
    pool = [RUST_PREDATOR, RUST_IMPROVED]  # deterministic bots only
    sizes = [None, "20 20", "25 25", "30 30", "35 35", "40 40", "50 50", "24 40", "40 24", "15 15"]
    cases = []
    count = 18 if quick else 72
    for k in range(count):
        n = k % 6 + 1
        bots = [rng.choice(pool) for _ in range(n)]
        size = rng.choice(sizes)
        seed = rng.randrange(2 ** 31)
        args = ["-s", str(seed)]
        if size:
            args += ["-d", size]
        if k % 4 == 0:
            args.append("-q")          # also compare the JSON summary
        if k % 5 == 1:
            args.append("--no-html")
        if k % 9 == 2:
            args += ["--turns", str(rng.randint(1, 80))]
        cases.append(("game%02d" % k, bots, args))
    return cases


# Test bots for the ejection checks, raw protocol so they need no imports.
BOT_SOURCE = r'''
import sys, zlib, random, time
MODE, TURN = sys.argv[1], int(sys.argv[2])

def readline():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return line.rstrip("\n")

tag = int(readline()); w, h = map(int, readline().split()); readline(); readline()
if MODE == "init-crash":
    sys.exit(1)
if MODE == "init-close":
    sys.stdout.close(); time.sleep(5); sys.exit(0)
names = {"garbage": "  Ünïcødé \"bot\" \\ 名前 \U0001F642 with a long name over 30 chars ",
         "blank": "   "}
end = "\r\n" if MODE == "garbage" else "\n"
sys.stdout.write(names.get(MODE, MODE + "Bot") + end); sys.stdout.flush()
turn = 0
while True:
    frame = readline(); turn += 1
    nums = list(map(int, frame.split()))
    owners, i = [], 0
    while len(owners) < w * h:
        owners += [nums[i + 1]] * nums[i]; i += 2
    rng = random.Random(zlib.crc32(frame.encode()) + tag)
    if turn == TURN:
        if MODE == "crash":
            sys.exit(3)
        if MODE == "timeout":
            time.sleep(3)
        if MODE == "close":      # stdout closed, process still running
            sys.stdout.close(); time.sleep(30)
        if MODE == "stdin-close":
            # Answers this turn twice, then stops reading: the next frame
            # fails with a closed pipe but the spare answer is still read.
            import os
            os.close(0)
            sys.stdout.write("0 0 0" + end + "0 0 0" + end); sys.stdout.flush(); time.sleep(30)
    moves = []
    for k, o in enumerate(owners):
        if o == tag or (MODE == "garbage" and rng.random() < 0.05):
            moves += [str(k % w), str(k // w), str(rng.choice([0, 0, 1, 2, 3, 4]))]
    if MODE == "garbage":
        junk = ["x", "-1", "999", "5", "+1", "1_0", "07", "0x1", "3.0", "1e2", "--2"]
        for _ in range(10):
            moves.insert(rng.randrange(len(moves) + 1), rng.choice(junk))
        moves += moves[:30]   # duplicates: first one wins
    sys.stdout.write(" ".join(moves) + end); sys.stdout.flush()
'''


def ejection_cases(tmp):
    path = os.path.join(tmp, "testbot.py")
    with open(path, "w") as f:
        f.write(BOT_SOURCE)
    tb = lambda mode, turn=0: "%s %s %s %d" % (sys.executable, path, mode, turn)
    return [
        ("crash", [RUST_IMPROVED, tb("crash", 20)], ["-s", "11", "-d", "20 20"]),
        ("timeout", [RUST_PREDATOR, tb("timeout", 15), RUST_IMPROVED], ["-s", "12", "-d", "24 20"]),
        ("close", [tb("close", 9), RUST_PREDATOR], ["-s", "13", "-d", "20 20"]),
        ("stdin-close", [tb("stdin-close", 6), RUST_IMPROVED], ["-s", "20", "-d", "20 20"]),
        ("init-crash", [tb("init-crash"), RUST_PREDATOR, RUST_IMPROVED], ["-s", "14", "-d", "21 21"]),
        ("init-close", [RUST_PREDATOR, tb("init-close")], ["-s", "15", "-d", "20 20"]),
        ("garbage", [tb("garbage"), RUST_PREDATOR, tb("blank")], ["-s", "16", "-d", "30 30"]),
        ("garbage-q", [tb("garbage"), tb("crash", 5), RUST_PREDATOR, tb("timeout", 7)],
         ["-s", "17", "-d", "30 30", "-q"]),
        ("solo", [tb("garbage")], ["-s", "18", "-d", "20 20", "--turns", "40"]),
        ("missing", ["/nonexistent/bot", RUST_PREDATOR], ["-s", "19", "-d", "20 20", "-q"]),
    ]


def check_games(cases, tmp, workers, title):
    failed = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(compare_game, label, bots, args, tmp) for label, bots, args in cases]
        for (label, bots, args), fut in zip(cases, futs):
            label, problems, turns = fut.result()
            names = ", ".join(os.path.basename(b.split()[-1]) if "testbot" not in b else b.split()[-2]
                              for b in bots)
            print("  %-4s %-11s %-30s %s turns  %s" % ("ok" if not problems else "FAIL", label,
                                                      " ".join(args), turns, names))
            for p in problems:
                print("       " + p.replace("\n", "\n       "))
            failed += bool(problems)
    if failed:
        fail("%s: %d of %d games differ" % (title, failed, len(cases)))
    print("%s: %d games identical" % (title, len(cases)))


def fail(msg):
    print("FAIL: " + msg)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="Check the Rust engine against halite.py")
    ap.add_argument("--quick", action="store_true", help="fewer cases")
    ap.add_argument("--only", choices=["maps", "rules", "games", "ejection"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    for path in [RUST_ENGINE, os.path.join(HERE, RUST_PREDATOR), os.path.join(HERE, RUST_IMPROVED)]:
        if not os.path.exists(path):
            fail("%s not built (cargo build --release)" % path)

    with tempfile.TemporaryDirectory(prefix="check_engine-") as tmp:
        if args.only in (None, "maps"):
            check_maps(args.quick)
        if args.only in (None, "rules"):
            check_rules(args.quick)
        if args.only in (None, "games"):
            check_games(game_cases(args.quick), tmp, args.workers, "games")
        if args.only in (None, "ejection"):
            # Fewer workers so a loaded machine does not cause a spurious timeout.
            check_games(ejection_cases(tmp), tmp, 3, "ejection")
    print("All checks passed: the Rust engine matches halite.py.")


if __name__ == "__main__":
    main()
