# Halite I clone

A local re-creation of **Halite**, the first season (2016 to 2017) of the AI programming
competition run by Two Sigma and Cornell Tech. Write a bot, pit it against others, and
watch the replay.

You need Python 3 (standard library only) for the tools and Rust for the bots. The bots and an
optional Rust engine use no external crates. Install Rust with [rustup](https://rustup.rs) if
needed: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`.

## Quick start

```bash
# Build the bots (and the faster engine, optional)
(cd starter-kit/rust && cargo build --release)
(cd engine && cargo build --release)

# Your bot vs the sample bot, random map size
python3 halite.py starter-kit/rust/target/release/my_bot starter-kit/rust/target/release/improved_bot

# Fixed size and seed, up to 6 players
python3 halite.py -d "30 30" -s 42 starter-kit/rust/target/release/my_bot starter-kit/rust/target/release/random_bot
```

Each game writes to `replays/`:

- a `.html` file: double-click it to watch the game (the replay is embedded)
- a `.hlt` file: the raw replay (JSON), which you can drop onto `viewer.html`

Whatever your bot writes to stderr is saved in `logs/bot-N.log`. Never print to stdout:
that is the channel to the game.

To generate a sample game to look at (from the repo root, after building):

```bash
B=starter-kit/rust/target/release
python3 halite.py -d "40 40" -s 2016 -o examples $B/predator_bot $B/improved_bot $B/improved_bot $B/random_bot
```

## Files

| File | What it is |
|---|---|
| `halite.py` | Game environment: map generator, rules, bot protocol, replays |
| `viewer.html` | Replay viewer (play, scrub, pan the wrap-around map, stats charts) |
| `tournament.py` | Runs many games and reports wins and average rank |
| `arena.py`, `arena.html` | Endless tournament server with a leaderboard and a browsable list of every game |
| `test_rules.py` | Checks for the combat, movement and production rules |
| `engine/` | Optional Rust engine (`halite` binary): same CLI, maps, rules and replays as `halite.py`, about 5x faster |
| `tools/` | Bot-building tools: `eval_bot.py` (win rates vs an opponent pool), `solo_bench.py` (expansion speed), `trace_game.py` (one game's economy over time) |
| `check_engine.py` | Proves the Rust engine plays exactly the same games as `halite.py` |
| `starter-kit/rust/` | Starter kit (no external crates): the `hlt` module and the bots `my_bot` (yours, starts random), `random_bot`, `improved_bot`, `predator_bot`, `apex_alpha` (strongest) |

## Options

```
python3 halite.py [-d "W H"] [-s SEED] [-t] [-q] [-o DIR] [--no-replay] [--no-html] [--turns N]
                  [--log-dir DIR] [--no-logs] BOT [BOT ...]
  -d   map size (default: random square from 20 to 50)
  -s   map seed, to replay the same map
  -t   no timeouts (lets you debug a slow bot)
  -q   print only a JSON summary
```

`engine/target/release/halite` takes exactly the same options.

```bash
B=starter-kit/rust/target/release
python3 tournament.py -n 20 $B/my_bot $B/improved_bot
python3 tournament.py -n 20 --players 4 $B/my_bot $B/random_bot $B/improved_bot $B/predator_bot ./other_bot
```

`tournament.py` and `arena.py` use the Rust engine when it is built and fall back to
`halite.py` otherwise; `--engine python` or `--engine rust` forces one.

## The arena

```bash
python3 arena.py
```

Then open http://localhost:8000. The arena plays games forever, several at once (half your CPU
cores by default, `--workers N` to change), between every bot in `starter-kit/rust/src/bin/`
that has been built: `improved_bot.rs` plays as ImprovedBot. A new bot joins on the next game once
`cargo build --release` has built it; a bot whose file is removed stays on the leaderboard as
"retired". The page shows an Elo-style leaderboard, the
games in progress and every finished game; click a bot to filter, click Watch to open a replay.
Charts: every bot's rating over time, a head-to-head matrix (how often each bot finishes ahead of
each other bot), and for a selected bot its toughest opponents, recent form (click a square to
watch that game) and win rate by number of players and by map size.
Games mix 2, 3, 4 and 6 players on random maps.

Bots outside the starter kit, in any language: anything that speaks the protocol works.

```bash
python3 arena.py --bot "CBot=./bots/c/cbot" --bot "OldBot=python3 my_2016_bot.py"
```

Everything is stored in `arena/`: `games.jsonl` holds the latest 10,000 games (`--history N`),
`state.json` the ratings and totals of everything older, plus the replays. Disk use, memory and
startup time stay bounded however long it runs. Stop with Ctrl-C and restart any time: nothing is lost. Only the latest 200 replays are kept (`--keep N`; a replay is about 2 MB).
The server listens on this machine only unless you pass `--host 0.0.0.0`.

## The rules

- The map is a grid that **wraps around** at the edges. Every site has a **production**
  value and a **strength** (0 to 255). Each player starts with one piece of strength 255.
- Every turn, each of your pieces either stays `STILL` or moves `NORTH`, `EAST`, `SOUTH` or `WEST`.
  All players move simultaneously.
- A piece that stays still gains its site's production. A piece that moves leaves behind
  a piece of yours with strength 0, so you keep the territory.
- Your pieces that end up on the same site merge. Strength is capped at 255 and any excess is lost.
- **Combat:** each of your pieces deals damage equal to its strength to every enemy piece on
  its site and on the 4 adjacent sites ("overkill"). All damage is dealt before any is taken,
  so one strong piece can hurt several enemies at once.
- **Neutral sites** only fight pieces that move onto them: capturing one costs its strength.
  They never damage, or get damaged by, pieces next to them.
- A piece whose strength falls to 0 or below from damage dies.
- The game ends when one player is left or after `10 × sqrt(width × height)` turns.
  Players are ranked by how long they survived, then by territory, then by strength.

## Bot protocol

Bots talk to the environment over stdin and stdout, one line per message, exactly as in the
original game, so any Halite I bot (in any language) should work.

**Init** (bot has 15 s to reply with its name):

```
<your player id>
<width> <height>
<production of every site, row by row>
<map>
```

**Each turn** (bot has 1 s to reply): the environment sends `<map>`, the bot replies with
`x y direction` triples for its pieces, all on one line, e.g. `3 4 1 3 5 0`.
Directions: 0 STILL, 1 NORTH, 2 EAST, 3 SOUTH, 4 WEST. Pieces without a move stay still.

**Map format:** run-length encoded owners as `count owner` pairs, covering the grid row by
row from the top-left, followed by the strength of every site in the same order.

A bot that crashes or times out is ejected and its sites become neutral.

## Writing a bot

`starter-kit/rust/` is a cargo project with no external crates:

```bash
cd starter-kit/rust
cargo build --release
cd ../..
python3 halite.py starter-kit/rust/target/release/my_bot starter-kit/rust/target/release/improved_bot
```

Edit `src/bin/my_bot.rs` (it starts as a random bot). To add another bot, drop a file in
`src/bin/`: `src/bin/foo.rs` builds to `target/release/foo`. The `hlt` module mirrors the
Python API in Rust style:

```rust
use hlt::*;

fn main() {
    let (my_id, _map, mut conn) = get_init();
    send_init("MyRustBot");
    loop {
        let map = conn.get_frame();
        let mut moves = Vec::new();
        for y in 0..map.height {
            for x in 0..map.width {
                let loc = Location { x, y };
                let site = map.get_site(loc, STILL); // .owner .strength .production
                if site.owner == my_id {
                    moves.push(Move { loc, direction: STILL });
                }
            }
        }
        conn.send_frame(&moves);
    }
}
```

Helpers on `GameMap`: `get_site(loc, dir)`, `get_location(loc, dir)`, `get_distance(a, b)`,
`get_angle(a, b)`, `in_bounds(loc)`; `map.contents[y][x]` is the raw grid. Never print to
stdout (it is the game channel); use `eprintln!`, which goes to `logs/bot-N.log`. The bot exits
cleanly when the environment closes stdin.

`improved_bot` is a good first target: it captures the best-value neighbouring site,
waits until it has 5x its production in strength, and sends interior pieces to the border.
Classic next steps from the original competition: route interior strength toward the
weakest point of the border, avoid merging past 255, and use overkill when fighting
(move into the empty gap between you and an enemy so your piece hits all their pieces at once).
`predator_bot` does several of these.

The bots were first written in Python; `improved_bot` and `predator_bot` are move-for-move ports
(identical replays on the same maps), and Rust versions run a game about 2.2x faster, so the
Python kit was retired. It is still in git history.

## The Rust engine

`engine/` is a Rust port of `halite.py` (standard library only). Same command line, same
console output, same `.hlt` and `.html` files, same bot protocol and timeouts, and above all
the same games: a seed gives the identical map (it reimplements Python's Mersenne Twister and
`random()`, `choice`, `shuffle`), and every turn gives the identical result. `halite.py`
stays the reference.

```bash
cd engine && cargo build --release && cd ..
engine/target/release/halite -d "30 30" -s 42 starter-kit/rust/target/release/my_bot starter-kit/rust/target/release/predator_bot
python3 check_engine.py          # prove it matches halite.py (about 2 minutes; --quick for less)
```

`check_engine.py` compares the two engines on thousands of maps (seeds, sizes, 1 to 6 players,
the default random size), on 20,000 random single-turn states plus the edge cases of
`test_rules.py` (zero-strength pieces, the 255 cap, overkill, neutral fights, tiny wrapping
maps), on 72 full games with deterministic bots, and on crash, timeout, closed-pipe, init-failure
and garbage-output bots. Replays, summaries and console output must match byte for byte.

Measured on a 50x50 4-player game with four Rust PredatorBots: 2.3 s on `halite.py`, 0.4 s on
the Rust engine. The arena with Rust bots goes from about 250 to about 1,370 games per minute.

One known difference: Python's `int()` accepts non-ASCII digits (such as Arabic-Indic `١`) in a
bot's move line, the Rust engine ignores such a move. No real bot sends those.

## How close is it to the original?

The rules follow the description in the creators' paper (Spector and Truell,
*The Design and Implementation of Modern Online Programming Competitions*, 2017) and the
original starter kit protocol. Two parts are re-implementations rather than copies:

- **Map generator:** smooth noise mirrored into one tile per player, with each start at its tile centre, so starts are fair and evenly spaced.
  Maps look and play similarly but are not byte-identical to the original generator.
  The map may shrink slightly so the tiles divide it evenly (e.g. 40x40 with 6 players is 39x40).
- **Edge cases** such as several zero-strength pieces of different players on one site
  (nobody keeps it) and ejected bots (their sites turn neutral).
