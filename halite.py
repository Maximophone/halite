#!/usr/bin/env python3
"""
Halite I clone: game environment.

A faithful re-creation of the 2016-17 Two Sigma / Cornell Tech Halite game.
Bots are separate programs that talk to this environment over stdin/stdout
using the original Halite I text protocol, so bots written for the original
starter kits should run unchanged.

Usage:
    python3 halite.py "python3 MyBot.py" "python3 ImprovedBot.py"
    python3 halite.py -d "40 40" -s 42 "python3 MyBot.py" "python3 RandomBot.py"

Run `python3 halite.py -h` for all options.
"""

import argparse
import datetime
import json
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STILL, NORTH, EAST, SOUTH, WEST = range(5)
MAX_STRENGTH = 255
START_STRENGTH = 255          # strength of each player's starting piece
INIT_TIMEOUT = 15.0           # seconds a bot gets to initialise and send its name
TURN_TIMEOUT = 1.0            # seconds a bot gets per turn
MAX_PLAYERS = 6
VERSION = 1

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Map generation
# ---------------------------------------------------------------------------

def _smooth_noise(w, h, rng, octaves=4, start_cells=2):
    """Multi-octave value noise in [0, 1]."""
    out = [[0.0] * w for _ in range(h)]
    amp = 1.0
    for octave in range(octaves):
        g = start_cells * (2 ** octave)
        grid = [[rng.random() for _ in range(g + 1)] for _ in range(g + 1)]
        for y in range(h):
            fy = y / h * g
            iy = int(fy)
            ty = fy - iy
            ty = ty * ty * (3 - 2 * ty)
            row_a, row_b = grid[iy], grid[iy + 1]
            out_row = out[y]
            for x in range(w):
                fx = x / w * g
                ix = int(fx)
                tx = fx - ix
                tx = tx * tx * (3 - 2 * tx)
                top = row_a[ix] + (row_a[ix + 1] - row_a[ix]) * tx
                bot = row_b[ix] + (row_b[ix + 1] - row_b[ix]) * tx
                out_row[x] += amp * (top + (bot - top) * ty)
        amp *= 0.5
    lo = min(min(r) for r in out)
    hi = max(max(r) for r in out)
    span = (hi - lo) or 1.0
    return [[(v - lo) / span for v in row] for row in out]


_LAYOUTS = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (1, 5), 6: (2, 3)}


def generate_map(width, height, num_players, rng):
    """
    Build a symmetric map. One tile is generated with smooth noise and then
    mirrored across a grid of tiles, one tile per player, so every player
    starts in an equivalent position.

    Returns (width, height, production, strength, starts) where width/height
    may be rounded down so the tiles divide the map evenly.
    """
    rows, cols = _LAYOUTS[num_players]
    if width < height and cols > rows:
        rows, cols = cols, rows
    tw, th = max(3, width // cols), max(3, height // rows)
    W, H = tw * cols, th * rows

    n_prod = _smooth_noise(tw, th, rng)
    n_str = _smooth_noise(tw, th, rng)

    tile_prod = [[0] * tw for _ in range(th)]
    tile_str = [[0] * tw for _ in range(th)]
    for y in range(th):
        for x in range(tw):
            p = n_prod[y][x]
            tile_prod[y][x] = int(round((p ** 2.2) * 15))
            s = 0.45 * n_str[y][x] + 0.55 * p
            tile_str[y][x] = int(round(min(MAX_STRENGTH, s ** 2.0 * 280 + 5)))

    # Starting position inside the tile, away from tile edges.
    sx = rng.randint(tw // 4, max(tw // 4, (3 * tw) // 4 - 1))
    sy = rng.randint(th // 4, max(th // 4, (3 * th) // 4 - 1))

    production = [[0] * W for _ in range(H)]
    strength = [[0] * W for _ in range(H)]
    starts = []
    for r in range(rows):
        for c in range(cols):
            flip_x, flip_y = c % 2 == 1, r % 2 == 1
            for y in range(th):
                ty = th - 1 - y if flip_y else y
                for x in range(tw):
                    tx = tw - 1 - x if flip_x else x
                    production[r * th + y][c * tw + x] = tile_prod[ty][tx]
                    strength[r * th + y][c * tw + x] = tile_str[ty][tx]
            px = c * tw + (tw - 1 - sx if flip_x else sx)
            py = r * th + (th - 1 - sy if flip_y else sy)
            starts.append((px, py))
    rng.shuffle(starts)
    return W, H, production, strength, starts[:num_players]


# ---------------------------------------------------------------------------
# Game state and rules
# ---------------------------------------------------------------------------

class GameState:
    def __init__(self, width, height, production, strength, starts):
        self.width = width
        self.height = height
        self.num_players = len(starts)
        self.production = production
        self.strength = [row[:] for row in strength]
        self.owner = [[0] * width for _ in range(height)]
        for i, (x, y) in enumerate(starts):
            self.owner[y][x] = i + 1
            self.strength[y][x] = START_STRENGTH

    def neighbor(self, x, y, d):
        if d == NORTH:
            return x, (y - 1) % self.height
        if d == SOUTH:
            return x, (y + 1) % self.height
        if d == EAST:
            return (x + 1) % self.width, y
        if d == WEST:
            return (x - 1) % self.width, y
        return x, y

    def serialize(self):
        """Original Halite I map format: run-length owners, then strengths."""
        parts = []
        count, current = 0, self.owner[0][0]
        for row in self.owner:
            for o in row:
                if o == current:
                    count += 1
                else:
                    parts.append(str(count))
                    parts.append(str(current))
                    count, current = 1, o
        parts.append(str(count))
        parts.append(str(current))
        for row in self.strength:
            parts.extend(str(s) for s in row)
        return " ".join(parts)

    def stats(self):
        n = self.num_players
        territory = [0] * (n + 1)
        strength = [0] * (n + 1)
        prod = [0] * (n + 1)
        for y in range(self.height):
            orow, srow, prow = self.owner[y], self.strength[y], self.production[y]
            for x in range(self.width):
                o = orow[x]
                territory[o] += 1
                strength[o] += srow[x]
                prod[o] += prow[x]
        return territory, strength, prod

    def step(self, moves):
        """
        Advance one turn. `moves` maps player -> {(x, y): direction}.
        Rules (Halite I):
          1. Still pieces gain their site's production. Moving pieces move
             and leave a zero-strength piece of the same owner behind.
          2. Pieces of one owner landing on the same site merge; strength is
             capped at 255 and the excess is lost.
          3. Every player piece deals damage equal to its strength to each
             enemy player piece on its own site and the 4 adjacent sites
             ("overkill"). Neutral sites only fight pieces on the same site,
             in both directions. All damage is dealt simultaneously.
          4. Pieces whose strength drops to zero or below from damage die.
             A surviving piece owns the site.
        """
        W, H, n = self.width, self.height, self.num_players
        owner, strength, production = self.owner, self.strength, self.production

        pieces = [dict() for _ in range(n + 1)]   # player -> {loc: strength}
        for y in range(H):
            for x in range(W):
                o = owner[y][x]
                if o == 0:
                    continue
                d = moves.get(o, {}).get((x, y), STILL)
                pmap = pieces[o]
                if d == STILL:
                    s = strength[y][x] + production[y][x]
                    pmap[(x, y)] = min(MAX_STRENGTH, pmap.get((x, y), 0) + s)
                else:
                    dest = self.neighbor(x, y, d)
                    pmap[dest] = min(MAX_STRENGTH, pmap.get(dest, 0) + strength[y][x])
                    pmap.setdefault((x, y), 0)

        damage = [dict() for _ in range(n + 1)]
        neutral_damage = {}
        for p in range(1, n + 1):
            for (x, y), s in pieces[p].items():
                if s > 0:
                    targets = [(x, y)] + [self.neighbor(x, y, d) for d in (NORTH, EAST, SOUTH, WEST)]
                    for q in range(1, n + 1):
                        if q == p:
                            continue
                        qmap, qdmg = pieces[q], damage[q]
                        for t in targets:
                            if t in qmap:
                                qdmg[t] = qdmg.get(t, 0) + s
                if owner[y][x] == 0:
                    # Player piece sitting on a neutral site: fights the neutral.
                    neutral_damage[(x, y)] = neutral_damage.get((x, y), 0) + s
                    ns = strength[y][x]
                    if ns > 0:
                        damage[p][(x, y)] = damage[p].get((x, y), 0) + ns

        new_owner = [[0] * W for _ in range(H)]
        new_strength = [[strength[y][x] if owner[y][x] == 0 else 0 for x in range(W)] for y in range(H)]

        # Neutral sites that were entered.
        for (x, y) in neutral_damage:
            remaining = strength[y][x] - neutral_damage[(x, y)]
            new_strength[y][x] = remaining if remaining > 0 else 0

        survivors = {}
        for p in range(1, n + 1):
            pdmg = damage[p]
            for loc, s in pieces[p].items():
                dmg = pdmg.get(loc, 0)
                remaining = s - dmg
                if remaining > 0 or (dmg == 0 and s == 0):
                    survivors.setdefault(loc, []).append((p, max(0, remaining)))

        for (x, y), lst in survivors.items():
            if len(lst) == 1:
                p, s = lst[0]
                if owner[y][x] == 0 and (x, y) in neutral_damage and new_strength[y][x] > 0:
                    continue  # neutral held the site (only reachable with zero-strength attackers)
                new_owner[y][x] = p
                new_strength[y][x] = s
            else:
                # Several zero-strength pieces of different owners: nobody holds it.
                new_owner[y][x] = 0
                new_strength[y][x] = 0

        self.owner, self.strength = new_owner, new_strength

    def eject(self, player):
        """Remove a crashed or timed-out bot: its sites become neutral."""
        for y in range(self.height):
            for x in range(self.width):
                if self.owner[y][x] == player:
                    self.owner[y][x] = 0


# ---------------------------------------------------------------------------
# Bot process handling
# ---------------------------------------------------------------------------

class Bot:
    def __init__(self, tag, command, log_path, no_timeout):
        self.tag = tag
        self.command = command
        self.name = command
        self.no_timeout = no_timeout
        self.alive = True
        self.error = None
        self.log = open(log_path, "w") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, bufsize=1)
        self.lines = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        try:
            for line in self.proc.stdout:
                self.lines.put(line.rstrip("\r\n"))
        except Exception:
            pass
        self.lines.put(None)

    def send(self, text):
        try:
            self.proc.stdin.write(text + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            self.fail("crashed (pipe closed)")

    def recv(self, deadline):
        timeout = None if self.no_timeout else max(0.0, deadline - time.time())
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            self.fail("timed out")
            return None
        if line is None:
            self.fail("crashed (exited)")
        return line

    def fail(self, reason):
        if self.alive:
            self.alive = False
            self.error = reason

    def kill(self):
        try:
            self.proc.kill()
        except Exception:
            pass
        if self.log is not subprocess.DEVNULL:
            self.log.close()


def parse_moves(line, state, player):
    moves = {}
    tokens = line.split()
    for i in range(0, len(tokens) - 2, 3):
        try:
            x, y, d = int(tokens[i]), int(tokens[i + 1]), int(tokens[i + 2])
        except ValueError:
            continue
        if 0 <= x < state.width and 0 <= y < state.height and 0 <= d <= 4:
            if state.owner[y][x] == player and (x, y) not in moves:
                moves[(x, y)] = d
    return moves


# ---------------------------------------------------------------------------
# Running a game
# ---------------------------------------------------------------------------

def run_game(commands, width=None, height=None, seed=None, no_timeout=False,
             replay_dir="replays", write_html=True, log_dir="logs", quiet=False,
             max_turns=None):
    n = len(commands)
    if not 1 <= n <= MAX_PLAYERS:
        raise ValueError("Halite supports 1 to %d players" % MAX_PLAYERS)
    if seed is None:
        seed = random.randrange(2 ** 31)
    rng = random.Random(seed)
    if width is None or height is None:
        side = rng.choice([20, 25, 30, 35, 40, 45, 50])
        width = width or side
        height = height or side

    W, H, production, strength, starts = generate_map(width, height, n, rng)
    state = GameState(W, H, production, strength, starts)
    turn_limit = max_turns or int(10 * math.sqrt(W * H))

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    bots = []
    for i, cmd in enumerate(commands):
        log_path = os.path.join(log_dir, "bot-%d.log" % (i + 1)) if log_dir else None
        bots.append(Bot(i + 1, cmd, log_path, no_timeout))

    def say(msg):
        if not quiet:
            print(msg, flush=True)

    say("Map %dx%d, seed %d, %d players, turn limit %d" % (W, H, seed, n, turn_limit))

    # --- init ---
    prod_line = " ".join(str(p) for row in production for p in row)
    map_line = state.serialize()
    for b in bots:
        b.send(str(b.tag))
        b.send("%d %d" % (W, H))
        b.send(prod_line)
        b.send(map_line)
    deadline = time.time() + INIT_TIMEOUT
    for b in bots:
        if b.alive:
            name = b.recv(deadline)
            if name is not None:
                b.name = name.strip()[:30] or b.command
    death_turn = [None] * (n + 1)
    for b in bots:
        if not b.alive:
            say("Player %d (%s) %s during init" % (b.tag, b.name, b.error))
            state.eject(b.tag)
            death_turn[b.tag] = 0

    frames = [(_flat(state.owner), _flat(state.strength))]
    move_frames = []
    turn = 0

    while turn < turn_limit:
        territory, _, _ = state.stats()
        for p in range(1, n + 1):
            if death_turn[p] is None and territory[p] == 0:
                death_turn[p] = turn
        active = [b for b in bots if b.alive and death_turn[b.tag] is None]
        if len([p for p in range(1, n + 1) if death_turn[p] is None]) <= (0 if n == 1 else 1):
            break

        turn += 1
        map_line = state.serialize()
        for b in active:
            b.send(map_line)
        all_moves = {}
        for b in active:
            line = b.recv(time.time() + TURN_TIMEOUT if not b.no_timeout else 0)
            if line is None:
                say("Turn %d: player %d (%s) %s and was ejected" % (turn, b.tag, b.name, b.error))
                state.eject(b.tag)
                death_turn[b.tag] = turn
                continue
            all_moves[b.tag] = parse_moves(line, state, b.tag)

        dirs = ["0"] * (W * H)
        for p, mv in all_moves.items():
            for (x, y), d in mv.items():
                dirs[y * W + x] = str(d)
        move_frames.append("".join(dirs))

        state.step(all_moves)
        frames.append((_flat(state.owner), _flat(state.strength)))

    territory, total_strength, _ = state.stats()
    for p in range(1, n + 1):
        if death_turn[p] is None and territory[p] == 0:
            death_turn[p] = turn

    for b in bots:
        b.kill()

    def rank_key(p):
        dt = death_turn[p]
        return (0 if dt is None else 1, -(dt or 0), -territory[p], -total_strength[p])

    order = sorted(range(1, n + 1), key=rank_key)
    results = []
    for rank, p in enumerate(order, 1):
        b = bots[p - 1]
        status = "alive" if death_turn[p] is None else "died turn %d" % death_turn[p]
        if b.error:
            status = "%s (%s)" % (status, b.error)
        results.append({"rank": rank, "player": p, "name": b.name, "command": b.command,
                        "territory": territory[p], "strength": total_strength[p],
                        "status": status})

    replay = {
        "version": VERSION, "width": W, "height": H, "num_players": n,
        "num_frames": len(frames), "seed": seed,
        "player_names": [b.name for b in bots],
        "productions": _flat(production),
        "frames": [[o, s] for o, s in frames],
        "moves": move_frames,
        "results": results,
    }

    replay_path = html_path = None
    if replay_dir:
        os.makedirs(replay_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        base = os.path.join(replay_dir, "%s-%d" % (stamp, seed))
        replay_path = base + ".hlt"
        with open(replay_path, "w") as f:
            json.dump(replay, f, separators=(",", ":"))
        if write_html:
            html_path = base + ".html"
            write_replay_html(replay, html_path)

    say("")
    say("Game over after %d turns." % turn)
    for r in results:
        say("  #%d  P%d %-24s territory %4d  strength %6d  %s" % (
            r["rank"], r["player"], r["name"], r["territory"], r["strength"], r["status"]))
    if replay_path:
        say("Replay: %s" % replay_path)
    if html_path:
        say("Viewer: %s" % html_path)

    return {"seed": seed, "width": W, "height": H, "turns": turn, "results": results,
            "replay": replay_path, "html": html_path}


def _flat(grid):
    return [v for row in grid for v in row]


def write_replay_html(replay, path):
    with open(os.path.join(HERE, "viewer.html")) as f:
        template = f.read()
    data = json.dumps(replay, separators=(",", ":")).replace("</", "<\\/")
    tag = "<script>window.EMBEDDED_REPLAY = %s;</script>" % data
    with open(path, "w") as f:
        f.write(template.replace("<!--EMBEDDED_REPLAY-->", tag))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Halite I clone game environment")
    ap.add_argument("bots", nargs="+", help='bot commands, e.g. "python3 MyBot.py"')
    ap.add_argument("-d", "--dimensions", help='map size as "WIDTH HEIGHT" (default: random 20 to 50)')
    ap.add_argument("-s", "--seed", type=int, help="map seed")
    ap.add_argument("-t", "--no-timeout", action="store_true", help="disable timeouts (for debugging)")
    ap.add_argument("-q", "--quiet", action="store_true", help="print only a JSON summary")
    ap.add_argument("-o", "--replay-dir", default="replays", help="where to write replays")
    ap.add_argument("--no-replay", action="store_true", help="do not write replay files")
    ap.add_argument("--no-html", action="store_true", help="skip the standalone HTML replay")
    ap.add_argument("--turns", type=int, help="override the turn limit")
    args = ap.parse_args()

    w = h = None
    if args.dimensions:
        parts = args.dimensions.replace("x", " ").split()
        w, h = int(parts[0]), int(parts[1])

    result = run_game(args.bots, w, h, args.seed, args.no_timeout,
                      None if args.no_replay else args.replay_dir,
                      not args.no_html, "logs", args.quiet, args.turns)
    if args.quiet:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
