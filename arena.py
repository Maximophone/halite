#!/usr/bin/env python3
"""
Halite arena: an endless tournament with a web front end.

Plays games forever between every bot it knows, several games in parallel,
keeps skill ratings (a mean and an uncertainty per bot), and serves a
leaderboard plus a browsable list of every game at http://localhost:8000
(each game opens in the replay viewer).

Bots: every Rust bot in starter-kit/rust/src/bin/ that has been built
(cargo build --release) is picked up automatically, including bots built while
the arena runs; improved_bot.rs plays as "ImprovedBot". Add other bots, in any
language, with --bot NAME=COMMAND.

Usage:
    python3 arena.py
    python3 arena.py --workers 8 --port 8000 --bot "CBot=./bots/c/cbot"

Data lives in arena/: games.jsonl (the latest --history games), state.json
(skills and totals for everything older) and replays/. Skills are rebuilt
from both on start, so stopping and restarting loses nothing.
"""

import argparse
import concurrent.futures as cf
import glob
import http.server
import json
import math
import os
import random
import shlex
import sys
import threading
import time
import urllib.parse

from halite import play_game, resolve_engine

HERE = os.path.dirname(os.path.abspath(__file__))
RUST_KIT = os.path.join(HERE, "starter-kit", "rust")


def bot_name(stem):
    """improved_bot -> ImprovedBot"""
    return "".join(part.capitalize() for part in stem.split("_"))


def rust_bots():
    """{name: binary} for every built bot, and the sources not built yet."""
    built, unbuilt = {}, []
    for src in sorted(glob.glob(os.path.join(RUST_KIT, "src", "bin", "*.rs"))):
        stem = os.path.splitext(os.path.basename(src))[0]
        binary = os.path.join(RUST_KIT, "target", "release", stem)
        if os.access(binary, os.X_OK):
            built[bot_name(stem)] = shlex.quote(binary)
        else:
            unbuilt.append(stem)
    return built, unbuilt

# Skill ratings: Weng-Lin Bayesian approximation with the Plackett-Luce model
# (Weng and Lin, JMLR 2011; what the openskill package implements), a
# patent-free cousin of TrueSkill that rates whole finishing orders. Each bot
# has a mean MU and an uncertainty SIGMA; the leaderboard ranks by
# MU - 3 * SIGMA, which a bot only climbs once it has played enough games.
MU = 25.0
SIGMA = MU / 3            # prior uncertainty of a new bot
BETA = MU / 6             # performance noise within one game
# Uncertainty added before each game, so ratings keep moving. The usual
# MU / 300 kept sigma near 1.7 here and the curves jittered by about 1.3;
# with MU / 3000 sigma settles near 0.7 and the jitter halves.
TAU = MU / 3000
KAPPA = 1e-4              # floor on the variance shrink factor
ANALYTICS_TTL = 5.0       # seconds the chart data is reused while games finish
COMPACT_SLACK = 1000      # games past --history before games.jsonl is rewritten
# Player counts and how often each is picked.
PLAYER_COUNTS = [(2, 0.5), (3, 0.1), (4, 0.3), (6, 0.1)]
# Halite matchmaking (see halite_lineup): the opponents' window is
# WINDOW_SHARE of the pool divided by u ** WINDOW_TAIL, u uniform.
WINDOW_SHARE = 1 / 6
WINDOW_TAIL = 0.65


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

def conservative(skill):
    mu, sigma = skill
    return mu - 3 * sigma


def rate(skills, ranks):
    """
    One Plackett-Luce update. `skills` are (mu, sigma) per seat, `ranks` the
    finishing places (1 = first, equal places are ties). Returns the new
    (mu, sigma) per seat.
    """
    n = len(skills)
    var = [sigma * sigma + TAU * TAU for _, sigma in skills]
    c = math.sqrt(sum(v + BETA * BETA for v in var))
    strength = [math.exp(mu / c) for mu, _ in skills]
    # For each seat q: the total strength of everyone placed at or below q,
    # and how many share q's place.
    below = [sum(strength[s] for s in range(n) if ranks[s] >= ranks[q]) for q in range(n)]
    tied = [sum(1 for s in range(n) if ranks[s] == ranks[q]) for q in range(n)]
    out = []
    for i, (mu, _) in enumerate(skills):
        omega = delta = 0.0
        for q in range(n):
            if ranks[q] > ranks[i]:
                continue
            p = strength[i] / below[q]
            omega += ((i == q) - p) / tied[q]
            delta += p * (1 - p) / tied[q]
        sd = math.sqrt(var[i])
        mu += var[i] / c * omega
        shrink = max(1 - sd / c * var[i] / (c * c) * delta, KAPPA)
        out.append((mu, sd * math.sqrt(shrink)))
    return out


def apply_game(game, skills, stats):
    """Update per-bot totals and skills with one finished game."""
    if game.get("reset_bot"):
        bot = game["reset_bot"]
        skills[bot] = [MU, SIGMA]
        stats[bot] = {"games": 0, "wins": 0, "score": 0.0, "errors": 0}
        return
    players = game["players"]
    n = len(players)
    for p in players:
        s = stats.setdefault(p["bot"], {"games": 0, "wins": 0, "score": 0.0, "errors": 0})
        s["games"] += 1
        s["wins"] += p["rank"] == 1
        s["score"] += (n - p["rank"]) / (n - 1) if n > 1 else 1.0
        s["errors"] += "(" in p["status"]
        skills.setdefault(p["bot"], [MU, SIGMA])
    if n < 2:
        return
    new = rate([skills[p["bot"]] for p in players], [p["rank"] for p in players])
    # A bot with several seats (fewer bots than seats) gets the sum of its
    # seats' mean changes and the product of their variance shrinks.
    change = {}
    for p, (mu, sigma) in zip(players, new):
        old_mu, old_sigma = skills[p["bot"]]
        d = change.setdefault(p["bot"], [0.0, 1.0])
        d[0] += mu - old_mu
        d[1] *= sigma * sigma / (old_sigma * old_sigma + TAU * TAU)
    for bot, (dmu, shrink) in change.items():
        mu, sigma = skills[bot]
        skills[bot] = [mu + dmu, math.sqrt((sigma * sigma + TAU * TAU) * shrink)]


def uniform_lineup(rng, names, n):
    """n seats from `names`, everyone seated once before anyone twice."""
    lineup = rng.sample(names, min(n, len(names)))
    while len(lineup) < n:
        lineup.append(rng.choice(names))
    rng.shuffle(lineup)
    return lineup


def halite_lineup(rng, names, n, skills, last_seen):
    """
    n seats chosen the way the Halite I servers did (ManagerAPI.php, task()).
    A seed bot first:
      - half the time at random, weighted by sigma^2 (uncertain bots, such as
        new ones, play more);
      - a quarter of the time one of the 3 bots that waited longest;
      - otherwise the bot that waited longest, so no bot is ever forgotten.
    Then the others at random among the K bots whose mu is closest to the
    seed's, K = WINDOW_SHARE * pool / u ** WINDOW_TAIL (u uniform). K is
    usually small, so most games are between near-equals, but its long tail
    still pairs distant bots now and then, which keeps the ratings of strong
    and weak bots on one scale. Halite used 5 / u ** 0.65 on ~1,500 bots;
    the window is scaled here since the pool is ~15.
    `skills`: bot -> (mu, sigma), new bots get the prior. `last_seen`: bot ->
    id of its last game (missing = never played).
    """
    def skill(b):
        return skills.get(b, (MU, SIGMA))
    stale = sorted(names, key=lambda b: (last_seen.get(b, 0), b))
    r = rng.random()
    if r < 0.5:
        seed = rng.choices(names, [skill(b)[1] ** 2 for b in names])[0]
    elif r < 0.75:
        seed = rng.choice(stale[:3])
    else:
        seed = stale[0]
    others = sorted((b for b in names if b != seed),
                    key=lambda b: (abs(skill(b)[0] - skill(seed)[0]), b))
    u = 1.0 - rng.random()  # in (0, 1]
    k = math.ceil(max(2.0, WINDOW_SHARE * len(names)) / u ** WINDOW_TAIL)
    window = others[:max(k, n - 1)]
    lineup = [seed] + rng.sample(window, min(n - 1, len(window)))
    while len(lineup) < n:  # fewer bots than seats
        lineup.append(rng.choice(names))
    rng.shuffle(lineup)
    return lineup


class Arena:
    def __init__(self, data_dir, extra_bots, discover, keep, history, engine="python",
                 matchmaking="halite"):
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
        self.matchmaking = matchmaking
        self.lock = threading.Lock()
        self.paused = False
        self.running = {}            # game id -> spec
        self.games = []              # summaries, oldest first
        self.skills = {}             # bot -> [mu, sigma]
        self.stats = {}
        self.last_seen = {}          # bot -> id of its last game, started or finished
        # Snapshot of everything older than the games kept in games.jsonl.
        self.base = {"skill": {}, "stats": {}, "games": 0, "max_id": 0, "last_fold_ids": []}
        self.started = time.time()
        self.played_since_start = 0
        self.next_id = 1
        self._load()

    def bots(self):
        found = dict(self.extra_bots)
        if self.discover:
            for name, cmd in rust_bots()[0].items():
                found.setdefault(name, cmd)
        return found

    def _load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                self.base = json.load(f)
        if "skill" not in self.base:
            # Snapshot from the Elo arena: its Elo ratings cannot be turned
            # into skills, so skills restart from the games still listed
            # (totals are kept).
            self.base.pop("ratings", None)
            self.base["skill"] = {}
        self.skills = {b: list(v) for b, v in self.base["skill"].items()}
        self.stats = {b: dict(s) for b, s in self.base["stats"].items()}
        self.resets = dict(self.base.get("resets", {}))
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
        """Add a finished game to the in-memory state and update skills."""
        self.games.append(game)
        apply_game(game, self.skills, self.stats)
        if game.get("reset_bot"):
            self.resets[game["reset_bot"]] = game["id"]
            self.last_seen.pop(game["reset_bot"], None)
        for p in game["players"]:
            self.last_seen[p["bot"]] = max(self.last_seen.get(p["bot"], 0), game["id"])

    def finish(self, game):
        with self.lock:
            self.running.pop(game["id"], None)
            with open(self.games_path, "a") as f:
                f.write(json.dumps(game, separators=(",", ":")) + "\n")
            self._record(game)
            self.played_since_start += 1
            self._prune()
            self._compact()

    def reset_bot(self, bot):
        """Start a new release without changing any other bot's accumulated results.

        The reset is an ordered journal event, so restarts and compaction replay
        old results for opponents before resetting only the requested identity.
        Historical entries for the previous release are hidden from the arena.
        """
        with self.lock:
            if not self.paused or self.running:
                raise ValueError("Pause the arena and let all running games finish first")
            if bot not in self.stats and bot not in self.bots():
                raise ValueError("Unknown bot: %s" % bot)
            event = {"id": self.next_id, "time": time.time(), "reset_bot": bot,
                     "players": [], "replay": None}
            with open(self.games_path, "a") as f:
                f.write(json.dumps(event, separators=(",", ":")) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.next_id += 1
            self._record(event)
            self._analytics = None
            return event

    def _visible_game(self, game):
        return not game.get("reset_bot") and all(
            game["id"] > self.resets.get(p["bot"], -1) for p in game["players"])

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
            apply_game(g, base["skill"], base["stats"])
            if g.get("reset_bot"):
                base.setdefault("resets", {})[g["reset_bot"]] = g["id"]
            if g.get("replay"):
                try:
                    os.remove(os.path.join(self.replay_dir, g["replay"]))
                except OSError:
                    pass
        base["games"] += sum(not g.get("reset_bot") for g in fold)
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
        with self.lock:
            if self.matchmaking == "halite":
                lineup = halite_lineup(rng, names, n, self.skills, self.last_seen)
            else:
                lineup = uniform_lineup(rng, names, n)
            gid = self.next_id
            self.next_id += 1
            for b in lineup:
                self.last_seen[b] = gid
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
            skills = {b: list(v) for b, v in self.base["skill"].items()}
            stats = {b: dict(v) for b, v in self.base["stats"].items()}
            resets = dict(self.resets)
            first = self.base["games"] + 1
        step = max(1, len(games) // 400)
        xs, curves = [], {}
        h2h = {}      # h2h[a][b] = [games where a finished ahead of b, games together]
        per = {}
        for k, g in enumerate(games):
            apply_game(g, skills, stats)
            if g.get("reset_bot"):
                continue
            players = [p for p in g["players"] if g["id"] > resets.get(p["bot"], -1)]
            n = len(g["players"])
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
                for bot, sk in skills.items():
                    if g["id"] <= resets.get(bot, -1):
                        continue
                    curves.setdefault(bot, [None] * (len(xs) - 1)).append(round(conservative(sk), 2))
                for bot, c in curves.items():
                    if len(c) < len(xs):
                        c.append(None)
        for b in per.values():
            b["recent"] = b["recent"][-60:]
        # Bots in order of first appearance ever, so chart colours never move.
        result = {"order": list(stats), "x": xs, "ratings": curves, "h2h": h2h, "bots": per,
                  "first_game": first, "games": len(games)}
        with self.lock:
            self._analytics = (key, time.time(), result)
        return result

    def state(self, limit, bot_filter):
        with self.lock:
            board = []
            active = self.bots()
            for bot, s in self.stats.items():
                mu, sigma = self.skills.get(bot, (MU, SIGMA))
                board.append({"bot": bot, "rating": round(mu - 3 * sigma, 2), "mu": round(mu, 2),
                              "sigma": round(sigma, 2), "games": s["games"],
                              "wins": s["wins"], "errors": s["errors"], "active": bot in active,
                              "score": round(s["score"] / s["games"], 3) if s["games"] else 0})
            # Bots still playing first; retired ones keep their final rating below.
            board.sort(key=lambda r: (not r["active"], -r["rating"]))
            games = [g for g in self.games if self._visible_game(g)]
            listed_games = len(games)
            if bot_filter:
                games = [g for g in games if any(p["bot"] == bot_filter for p in g["players"])]
            elapsed = max(1.0, time.time() - self.started)
            return {
                "paused": self.paused,
                "total_games": self.base["games"] + sum(not g.get("reset_bot") for g in self.games),
                "listed_games": listed_games,
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
                    help="add a bot that is not in starter-kit/rust (repeatable)")
    ap.add_argument("--no-discover", action="store_true", help="only use --bot entries")
    ap.add_argument("--keep", type=int, default=200, help="replays kept on disk (older ones are deleted)")
    ap.add_argument("--history", type=int, default=10000,
                    help="games listed and kept in games.jsonl; older ones only live on in the ratings")
    ap.add_argument("--data", default=os.path.join(HERE, "arena"), help="where games and replays are stored")
    ap.add_argument("--seed", type=int, help="seed for matchmaking")
    ap.add_argument("--matchmaking", choices=["halite", "uniform"], default="halite",
                    help="halite: seed bot by uncertainty or waiting time, opponents close in "
                         "skill (default); uniform: every lineup equally likely")
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
    arena = Arena(args.data, extra, not args.no_discover, args.keep, args.history, engine,
                  args.matchmaking)
    unbuilt = rust_bots()[1] if not args.no_discover else []
    if unbuilt:
        print("Not built, so not playing: %s (run `cargo build --release` in starter-kit/rust)"
              % ", ".join(unbuilt), flush=True)
    if len(arena.bots()) < 2:
        print("Need at least 2 bots, found: %s" % (", ".join(arena.bots()) or "none"))
        sys.exit(1)

    server = http.server.ThreadingHTTPServer((args.host, args.port), make_handler(arena))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("Arena running at http://%s:%d  (%d workers, %s engine, %s matchmaking, bots: %s)" % (
        "localhost" if args.host == "127.0.0.1" else args.host, args.port,
        args.workers, engine, args.matchmaking, ", ".join(sorted(arena.bots()))), flush=True)
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
