#!/usr/bin/env python3
"""
Halite arena: an endless tournament with a web front end.

Plays games forever between every bot it knows, several games in parallel,
keeps Elo-style ratings, and serves a leaderboard plus a browsable list of
every game at http://localhost:8000 (each game opens in the replay viewer).

Bots: every starter-kit/python/*Bot.py is picked up automatically, including
files added while the arena runs. Add other bots with --bot NAME=COMMAND.

Usage:
    python3 arena.py
    python3 arena.py --workers 8 --port 8000 --bot "RustBot=./target/release/mybot"

Data lives in arena/: games.jsonl (the latest --history games), state.json
(ratings and totals for everything older) and replays/. Ratings are rebuilt
from both on start, so stopping and restarting loses nothing.
"""

import argparse
import concurrent.futures as cf
import glob
import http.server
import json
import os
import random
import shlex
import sys
import threading
import time
import urllib.parse

from halite import play_game, resolve_engine

HERE = os.path.dirname(os.path.abspath(__file__))
BOT_GLOB = os.path.join(HERE, "starter-kit", "python", "*Bot.py")

START_RATING = 1500.0
K_FACTOR = 32.0
ANALYTICS_TTL = 5.0       # seconds the chart data is reused while games finish
COMPACT_SLACK = 1000      # games past --history before games.jsonl is rewritten
# Player counts and how often each is picked.
PLAYER_COUNTS = [(2, 0.5), (3, 0.1), (4, 0.3), (6, 0.1)]


def _write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Games (runs in worker processes)
# ---------------------------------------------------------------------------

def play(spec):
    """Run one game. `spec` is picklable so it can go to a worker process."""
    t0 = time.time()
    res = play_game([cmd for _, cmd in spec["bots"]], seed=spec["seed"],
                    replay_dir=spec["replay_dir"], write_html=False,
                    log_dir=None, engine=spec["engine"])
    players = []
    for r in res["results"]:
        bot_id = spec["bots"][r["player"] - 1][0]
        players.append({"bot": bot_id, "name": r["name"], "rank": r["rank"],
                        "territory": r["territory"], "strength": r["strength"],
                        "status": r["status"]})
    players.sort(key=lambda p: p["rank"])
    return {"id": spec["id"], "time": t0, "duration": round(time.time() - t0, 1),
            "seed": res["seed"], "width": res["width"], "height": res["height"],
            "turns": res["turns"], "players": players,
            "replay": os.path.basename(res["replay"])}


# ---------------------------------------------------------------------------
# Arena state
# ---------------------------------------------------------------------------

def apply_game(game, ratings, stats):
    """Update per-bot totals and ratings with one finished game."""
    players = game["players"]
    n = len(players)
    for p in players:
        s = stats.setdefault(p["bot"], {"games": 0, "wins": 0, "score": 0.0, "errors": 0})
        s["games"] += 1
        s["wins"] += p["rank"] == 1
        s["score"] += (n - p["rank"]) / (n - 1) if n > 1 else 1.0
        s["errors"] += "(" in p["status"]
        ratings.setdefault(p["bot"], START_RATING)
    # Multiplayer Elo: every pair of different bots is a duel won by the
    # better-ranked one, with K split over the n - 1 opponents.
    delta = {}
    for a in players:
        for b in players:
            if a["rank"] >= b["rank"] or a["bot"] == b["bot"]:
                continue
            ra, rb = ratings[a["bot"]], ratings[b["bot"]]
            expected = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
            change = K_FACTOR * (1.0 - expected) / (n - 1)
            delta[a["bot"]] = delta.get(a["bot"], 0.0) + change
            delta[b["bot"]] = delta.get(b["bot"], 0.0) - change
    for bot, d in delta.items():
        ratings[bot] += d


class Arena:
    def __init__(self, data_dir, extra_bots, discover, keep, history, engine="python"):
        self.data_dir = data_dir
        self.replay_dir = os.path.join(data_dir, "replays")
        self.games_path = os.path.join(data_dir, "games.jsonl")
        self.state_path = os.path.join(data_dir, "state.json")
        os.makedirs(self.replay_dir, exist_ok=True)
        self.extra_bots = extra_bots
        self.discover = discover
        self.keep = keep
        self.history = history
        self.engine = engine
        self.lock = threading.Lock()
        self.paused = False
        self.running = {}            # game id -> spec
        self.games = []              # summaries, oldest first
        self.ratings = {}
        self.stats = {}
        # Snapshot of everything older than the games kept in games.jsonl.
        self.base = {"ratings": {}, "stats": {}, "games": 0, "max_id": 0, "last_fold_ids": []}
        self.started = time.time()
        self.played_since_start = 0
        self.next_id = 1
        self._load()

    def bots(self):
        found = dict(self.extra_bots)
        if self.discover:
            for path in sorted(glob.glob(BOT_GLOB)):
                name = os.path.splitext(os.path.basename(path))[0]
                found.setdefault(name, "%s %s" % (shlex.quote(sys.executable), shlex.quote(path)))
        return found

    def _load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                self.base = json.load(f)
        self.ratings = dict(self.base["ratings"])
        self.stats = {b: dict(s) for b, s in self.base["stats"].items()}
        # Games folded into the snapshot just before a crash may still be in
        # games.jsonl: skip them.
        folded = set(self.base["last_fold_ids"])
        if os.path.exists(self.games_path):
            with open(self.games_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            game = json.loads(line)
                            if game["id"] not in folded:
                                self._record(game)
                        except (ValueError, KeyError):
                            pass  # a line cut short by a crash
        self.next_id = max([self.base["max_id"]] + [g["id"] for g in self.games]) + 1
        self._compact(force=True)  # in case --history is lower than last time
        self._prune()  # in case --keep is lower than last time
        # Replays of games cut off by the last shutdown were never recorded.
        known = {g["replay"] for g in self.games if g.get("replay")}
        for name in os.listdir(self.replay_dir):
            if name.endswith(".hlt") and name not in known:
                os.remove(os.path.join(self.replay_dir, name))

    def _record(self, game):
        """Add a finished game to the in-memory state and update ratings."""
        self.games.append(game)
        apply_game(game, self.ratings, self.stats)

    def finish(self, game):
        with self.lock:
            self.running.pop(game["id"], None)
            with open(self.games_path, "a") as f:
                f.write(json.dumps(game, separators=(",", ":")) + "\n")
            self._record(game)
            self.played_since_start += 1
            self._prune()
            self._compact()

    def _compact(self, force=False):
        """
        Fold the oldest games into state.json once games.jsonl holds more than
        `history` games (plus some slack, so the file is not rewritten after
        every game). Ratings stay exact: the snapshot is the state after the
        folded games, and the kept games are replayed on top of it.
        """
        extra = len(self.games) - self.history
        if extra <= 0 or (not force and extra < COMPACT_SLACK):
            return
        fold, keep = self.games[:extra], self.games[extra:]
        base = self.base
        for g in fold:
            apply_game(g, base["ratings"], base["stats"])
            if g.get("replay"):
                try:
                    os.remove(os.path.join(self.replay_dir, g["replay"]))
                except OSError:
                    pass
        base["games"] += len(fold)
        base["max_id"] = max([base["max_id"]] + [g["id"] for g in fold])
        base["last_fold_ids"] = [g["id"] for g in fold]
        # Snapshot first, then the shorter history. A crash in between leaves
        # folded games in games.jsonl, and _load skips them by id.
        _write_atomic(self.state_path, json.dumps(base, separators=(",", ":")))
        _write_atomic(self.games_path, "".join(
            json.dumps(g, separators=(",", ":")) + "\n" for g in keep))
        self.games = keep

    def _prune(self):
        with_replay = [g for g in self.games if g.get("replay")]
        for g in with_replay[:max(0, len(with_replay) - self.keep)]:
            try:
                os.remove(os.path.join(self.replay_dir, g["replay"]))
            except OSError:
                pass
            g["replay"] = None

    def new_spec(self, rng):
        bots = self.bots()
        if not bots:
            return None
        counts, weights = zip(*PLAYER_COUNTS)
        n = rng.choices(counts, weights)[0]
        names = list(bots)
        # Everyone gets a seat before any bot gets a second one.
        lineup = rng.sample(names, min(n, len(names)))
        while len(lineup) < n:
            lineup.append(rng.choice(names))
        rng.shuffle(lineup)
        with self.lock:
            gid = self.next_id
            self.next_id += 1
            spec = {"id": gid, "seed": rng.randrange(2 ** 31), "replay_dir": self.replay_dir,
                    "engine": self.engine,
                    "bots": [(b, bots[b]) for b in lineup], "started": time.time()}
            self.running[gid] = spec
        return spec

    def analytics(self):
        """
        Chart data over the games still listed (the latest --history), cached
        for a few seconds: rating curves, head-to-head counts and per-bot
        breakdowns. Ratings start from the snapshot of older games.
        """
        with self.lock:
            key = self.base["games"] + len(self.games)
            cached = getattr(self, "_analytics", None)
            if cached and (cached[0] == key or time.time() - cached[1] < ANALYTICS_TTL):
                return cached[2]
            games = list(self.games)
            ratings = dict(self.base["ratings"])
            stats = {b: dict(v) for b, v in self.base["stats"].items()}
            first = self.base["games"] + 1
        step = max(1, len(games) // 400)
        xs, curves = [], {}
        h2h = {}      # h2h[a][b] = [games where a finished ahead of b, games together]
        per = {}
        for k, g in enumerate(games):
            apply_game(g, ratings, stats)
            players = g["players"]
            n = len(players)
            for a in players:
                for b in players:
                    if a["bot"] == b["bot"]:
                        continue
                    cell = h2h.setdefault(a["bot"], {}).setdefault(b["bot"], [0, 0])
                    cell[0] += a["rank"] < b["rank"]
                    cell[1] += 1
            side = max(g["width"], g["height"])
            size = "20-25" if side <= 25 else "26-35" if side <= 35 else "36-45" if side <= 45 else "46-50"
            seen = set()
            for p in players:
                b = per.setdefault(p["bot"], {"by_players": {}, "by_size": {}, "recent": [], "turns_win": [0, 0], "turns_loss": [0, 0]})
                if p["bot"] in seen:
                    continue  # count a bot once per game (its best seat)
                seen.add(p["bot"])
                win = p["rank"] == 1
                for bucket, label in ((b["by_players"], str(n)), (b["by_size"], size)):
                    c = bucket.setdefault(label, [0, 0])
                    c[0] += win
                    c[1] += 1
                t = b["turns_win"] if win else b["turns_loss"]
                t[0] += g["turns"]
                t[1] += 1
                b["recent"].append({"id": g["id"], "rank": p["rank"], "n": n, "replay": g.get("replay"),
                                    "rivals": [q["bot"] for q in players if q is not p]})
            if k % step == 0 or k == len(games) - 1:
                xs.append(first + k)
                for bot, r in ratings.items():
                    curves.setdefault(bot, [None] * (len(xs) - 1)).append(round(r, 1))
                for bot, c in curves.items():
                    if len(c) < len(xs):
                        c.append(None)
        for b in per.values():
            b["recent"] = b["recent"][-60:]
        # Bots in order of first appearance ever, so chart colours never move.
        result = {"order": list(ratings), "x": xs, "ratings": curves, "h2h": h2h, "bots": per,
                  "first_game": first, "games": len(games)}
        with self.lock:
            self._analytics = (key, time.time(), result)
        return result

    def state(self, limit, bot_filter):
        with self.lock:
            board = []
            for bot, s in self.stats.items():
                board.append({"bot": bot, "rating": round(self.ratings[bot]), "games": s["games"],
                              "wins": s["wins"], "errors": s["errors"],
                              "score": round(s["score"] / s["games"], 3) if s["games"] else 0})
            board.sort(key=lambda r: -r["rating"])
            games = self.games
            if bot_filter:
                games = [g for g in games if any(p["bot"] == bot_filter for p in g["players"])]
            elapsed = max(1.0, time.time() - self.started)
            return {
                "paused": self.paused,
                "total_games": self.base["games"] + len(self.games),
                "listed_games": len(self.games),
                "matching_games": len(games),
                "games_per_min": round(self.played_since_start * 60.0 / elapsed, 1),
                "known_bots": sorted(self.bots()),
                "leaderboard": board,
                "running": [{"id": s["id"], "bots": [b for b, _ in s["bots"]],
                             "since": round(time.time() - s["started"])}
                            for s in sorted(self.running.values(), key=lambda s: s["id"])],
                "games": list(reversed(games[-limit:])),
            }


# ---------------------------------------------------------------------------
# Web front end
# ---------------------------------------------------------------------------

def make_handler(arena):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path, ctype):
            try:
                with open(path, "rb") as f:
                    self._send(200, f.read(), ctype)
            except OSError:
                self._send(404, b"not found", "text/plain")

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                self._file(os.path.join(HERE, "arena.html"), "text/html; charset=utf-8")
            elif url.path == "/viewer.html":
                self._file(os.path.join(HERE, "viewer.html"), "text/html; charset=utf-8")
            elif url.path == "/api/analytics":
                self._send(200, json.dumps(arena.analytics(), separators=(",", ":")).encode(), "application/json")
            elif url.path == "/api/state":
                limit = min(1000, int(q.get("limit", ["100"])[0]))
                body = json.dumps(arena.state(limit, q.get("bot", [""])[0])).encode()
                self._send(200, body, "application/json")
            elif url.path.startswith("/replays/"):
                name = os.path.basename(url.path)
                if name.endswith(".hlt"):
                    self._file(os.path.join(arena.replay_dir, name), "application/json")
                else:
                    self._send(404, b"not found", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path in ("/api/pause", "/api/resume"):
                arena.paused = self.path == "/api/pause"
                self._send(200, b'{"ok":true}', "application/json")
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Endless Halite tournament with a web front end")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="address to listen on (default: this machine only)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="games played in parallel (default: half the CPU cores)")
    ap.add_argument("--bot", action="append", default=[], metavar="NAME=COMMAND",
                    help="add a bot that is not in starter-kit/python (repeatable)")
    ap.add_argument("--no-discover", action="store_true", help="only use --bot entries")
    ap.add_argument("--keep", type=int, default=200, help="replays kept on disk (older ones are deleted)")
    ap.add_argument("--history", type=int, default=10000,
                    help="games listed and kept in games.jsonl; older ones only live on in the ratings")
    ap.add_argument("--data", default=os.path.join(HERE, "arena"), help="where games and replays are stored")
    ap.add_argument("--seed", type=int, help="seed for matchmaking")
    ap.add_argument("--engine", choices=["auto", "python", "rust"], default="auto",
                    help="game engine (default: Rust when engine/ is built, else Python)")
    args = ap.parse_args()

    extra = {}
    for entry in args.bot:
        name, sep, cmd = entry.partition("=")
        if not sep or not name or not cmd:
            ap.error("--bot expects NAME=COMMAND, got %r" % entry)
        extra[name.strip()] = cmd.strip()

    try:
        engine = resolve_engine(args.engine)
    except RuntimeError as err:
        ap.error(str(err))
    arena = Arena(args.data, extra, not args.no_discover, args.keep, args.history, engine)
    if len(arena.bots()) < 2:
        print("Need at least 2 bots, found: %s" % (", ".join(arena.bots()) or "none"))
        sys.exit(1)

    server = http.server.ThreadingHTTPServer((args.host, args.port), make_handler(arena))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("Arena running at http://%s:%d  (%d workers, %s engine, bots: %s)" % (
        "localhost" if args.host == "127.0.0.1" else args.host, args.port,
        args.workers, engine, ", ".join(sorted(arena.bots()))), flush=True)
    if arena.games:
        print("Resuming with %d games already played." % len(arena.games), flush=True)

    rng = random.Random(args.seed)
    pending = set()
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        try:
            while True:
                while not arena.paused and len(pending) < args.workers:
                    spec = arena.new_spec(rng)
                    if spec is None:
                        break
                    fut = pool.submit(play, spec)
                    fut.spec = spec
                    pending.add(fut)
                if not pending:
                    time.sleep(0.5)
                    continue
                done, pending = cf.wait(pending, timeout=0.5, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    try:
                        game = fut.result()
                    except Exception as err:  # engine error: report and carry on
                        with arena.lock:
                            arena.running.pop(fut.spec["id"], None)
                        print("Game %d failed: %s" % (fut.spec["id"], err), flush=True)
                        continue
                    arena.finish(game)
                    winner = game["players"][0]["bot"]
                    print("Game %d: %s won (%s, %dx%d, %d turns, %.0fs)" % (
                        game["id"], winner, " vs ".join(p["bot"] for p in game["players"]),
                        game["width"], game["height"], game["turns"], game["duration"]), flush=True)
        except KeyboardInterrupt:
            print("\nStopping: waiting for running games is skipped; they will not be recorded.")
            for fut in pending:
                fut.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            server.shutdown()


if __name__ == "__main__":
    main()
