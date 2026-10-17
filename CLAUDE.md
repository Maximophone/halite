# CLAUDE.md: context for working on this repo

This file holds everything a new session needs. README.md is the user-facing doc
(rules, protocol, how to play); this file covers history, design decisions and how
to work on the code.

## What this is and why

Maxime took part in Two Sigma's Halite AI programming competition years ago and asked
for a clone. "Around 2018" could have meant Halite II (space, 2017-18) or Halite III
(sea, 2018-19); he confirmed it was **Halite I** (Oct 2016 to Jan 2017, grid territory
game by Benjamin Spector and Michael Truell). He chose a **local engine + starter kit**
(like the original setup) over an in-browser bot editor.

Goal: faithful to the original rules and bot protocol, so a 2016-era bot runs unchanged.

Sources used for the rules:
- Spector and Truell, "The Design and Implementation of Modern Online Programming
  Competitions" (2017), https://arxiv.org/abs/1710.07738 (section 3.1.1 has the rules)
- https://en.wikipedia.org/wiki/Halite_AI_Programming_Competition
- Original repo: https://github.com/HaliteChallenge/Halite (not fetched; protocol and
  hlt/networking API were reproduced from knowledge of the 2016 Python starter kit)

## Layout

| Path | Role |
|---|---|
| `halite.py` | Everything engine-side: map generator, `GameState` (rules), `Bot` (subprocess + pipes), `run_game()`, replay writing, CLI |
| `viewer.html` | Self-contained replay viewer (canvas, no dependencies). Contains the placeholder `<!--EMBEDDED_REPLAY-->` that `write_replay_html()` replaces with `<script>window.EMBEDDED_REPLAY = {...}</script>` |
| `tournament.py` | Imports `run_game` and plays N games, prints wins and average rank |
| `arena.py` | Endless tournament: `ProcessPoolExecutor` runs `run_game` in parallel, results appended to `arena/games.jsonl`, multiplayer Elo rebuilt on start from `state.json` (snapshot of all games older than `--history`, default 10000) plus `games.jsonl`; `_compact` folds old games into the snapshot every `COMPACT_SLACK` games and is crash-safe via `last_fold_ids`; stdlib HTTP server for `arena.html`, `/api/state`, `/replays/*.hlt`, pause/resume |
| `arena.html` | Leaderboard + game list, polls `/api/state` every 3 s; links to `viewer.html?replay=/replays/x.hlt` |
| `test_rules.py` | Plain-assert rule checks (no pytest needed). Run `python3 test_rules.py` |
| `starter-kit/python/hlt.py`, `networking.py` | Original 2016 API (camelCase: `getInit`, `sendInit`, `getFrame`, `sendFrame`, `GameMap.getSite`, `Location`, `Move`, `STILL..WEST`) |
| `starter-kit/python/MyBot.py` | User's bot, starts as a random bot |
| `starter-kit/python/RandomBot.py` | Original sample bot |
| `starter-kit/python/ImprovedBot.py` | Heuristic sparring bot; beats RandomBot every time |
| `starter-kit/python/PredatorBot.py` | Strongest bot so far (see below) |
| `starter-kit/rust/` | Cargo project, no crates. `src/hlt.rs` is the lib (`hlt`: types plus `get_init`/`send_init`/`Connection::get_frame`/`send_frame`), `src/bin/my_bot.rs` random template, `src/bin/predator_bot.rs` exact port of PredatorBot (name `RustPredatorBot`) |

Generated, git-ignored: `starter-kit/rust/target/` (cargo build output), `arena/` (arena games and replays), `replays/` (`.hlt` + `.html` per game), `logs/bot-N.log` (bot stderr),
`examples/` (a sample game; regenerate with the command in README).

Constraints: **Python 3 standard library only** (tested on 3.10 on the Mac and 3.13 in the
cloud). No packages, no build step. Keep it that way unless asked.

## Rules as implemented (`GameState.step`)

Order within a turn:
1. Unspecified moves default to STILL. STILL: strength + production. Move: piece goes to the
   neighbour site, and a strength-0 piece of the same owner stays on the origin (territory kept).
2. Same-owner pieces on one site merge, `min(255, sum)`; excess lost.
3. Damage, all computed from post-merge strengths, applied simultaneously:
   - each player piece with strength > 0 deals its full strength to every enemy *player*
     piece on its site and the 4 neighbours (overkill, no splitting);
   - neutral sites interact only with player pieces on the *same* site, both ways;
     neutral never deals or receives adjacent damage.
4. A piece survives if `strength - damage > 0`, or if it had 0 strength and took 0 damage
   (this keeps left-behind zero pieces alive). Survivor owns the site with remaining strength.
   A neutral site keeps `max(0, strength - damage)` and stays neutral if no player survives.

Decisions where the paper is silent (my choices, documented in README too):
- Starting piece strength `START_STRENGTH = 255`.
- Several surviving zero-strength pieces of different owners on one site: site becomes neutral 0.
- Duplicate moves for one site in a bot's reply: first one wins. Moves for sites the bot
  doesn't own, out-of-range coordinates or bad directions are ignored.
- Timeouts: `INIT_TIMEOUT = 15 s` (name reply), `TURN_TIMEOUT = 1 s`. `-t` disables them.
- A bot that crashes, closes its pipe or times out is ejected: its sites become neutral
  (owner 0, strengths kept) and it is recorded as dying that turn.
- Turn limit `int(10 * sqrt(W * H))`. Game ends early when at most one player has territory
  (single-player games run to the limit).
- Ranking: survivors first, then later death is better, then territory, then total strength.

## Map generator (`generate_map`)

Not the original algorithm (that was not reproduced). Approach: one tile of multi-octave
value noise (`_smooth_noise`), mirrored across a grid of tiles, one per player
(`_LAYOUTS`: 2 -> 1x2, 3 -> 1x3, 4 -> 2x2, 5 -> 1x5, 6 -> 2x3; transposed for tall maps).
Odd-indexed tile rows/columns are flipped so edges join smoothly; every player's start is the
mirror image of the others. Starts sit at the tile centre: an earlier random offset meant flipped
neighbours could spawn 7 sites apart across the wrap seam on a 30x30 (a 9-turn rush game).
`test_starts_evenly_spaced` checks the nearest rival is at least one tile size minus 1 away. Width and height are rounded down to multiples of the tile size
(e.g. 40x40 with 6 players becomes 39x40, 25x25 with 2 players becomes 24x25).

Current tuning (changed once after playtesting):
- production `round(p ** 2.2 * 15)` -> 0..15, mean about 4.7
- strength `round(min(255, s ** 2 * 280 + 5))` with `s = 0.45 * noise + 0.55 * p`,
  so strength correlates with production; mean about 78, max about 180 on a 40x40.
- First version used `s ** 1.4 * 255`: neutrals were so strong that expansion stalled for
  ~150 turns. If you retune, check that two ImprovedBots visibly expand within ~50 turns.
Default map size when `-d` is omitted: random square from {20, 25, ..., 50}, seeded.

## Bot protocol (must stay compatible with original Halite I bots)

Init, one line each: `player_tag`, `W H`, all productions row-major, the map. Bot replies
with its name (truncated to 30 chars). Each turn: engine sends the map; bot replies one line
of `x y dir` triples (dir 0 STILL, 1 NORTH, 2 EAST, 3 SOUTH, 4 WEST). Map serialization:
run-length `count owner` pairs over the grid row-major from top-left, then W*H strengths.
Bots run with `shell=True` from the caller's cwd; stdout is the game channel, stderr goes to
`logs/bot-N.log`. `networking.getString` exits cleanly when stdin closes.

## Replay format (`.hlt`, JSON)

```
{ version: 1, width, height, num_players, num_frames, seed,
  player_names: [..],
  productions: [W*H ints, row-major],
  frames: [ [owners W*H], [strengths W*H] ] x num_frames   (frame 0 = initial state),
  moves: [ string of W*H digits 0-4 ] x (num_frames - 1)   (moves[f] goes from frame f to f+1),
  results: [ {rank, player, name, command, territory, strength, status} ] }
```
The viewer computes territory/production/strength curves from frames. A 40x40, 400-turn
4-player replay is about 3.7 MB (embedded HTML the same).

## How to check changes

```bash
python3 test_rules.py                                   # 10 rule checks, must all pass
python3 halite.py -d "30 30" -s 7 "python3 starter-kit/python/ImprovedBot.py" "python3 starter-kit/python/RandomBot.py"
python3 tournament.py -n 6 -s 1 "python3 starter-kit/python/ImprovedBot.py" "python3 starter-kit/python/RandomBot.py"
```
Already verified: crash and timeout ejection (a bot sleeping 2 s on turn 20, a bot exiting on
turn 2), 6-player games, viewer renders with no console errors (checked with a headless
Chromium screenshot). When you change rules, add a case to `test_rules.py`.

## PredatorBot (Maxime asked for "a predator": aggressive, combat-first)

Max-value field propagated inward from the frontier with Dijkstra (`DECAY` per step); interior
pieces follow it once grown (`WAIT` x production, `WAIT_FRONT` if the source is a fight). Fights
(enemy site, or 0-strength neutral next to an enemy) score `FIGHT_VALUE + heat/255`, where heat is
the enemy strength in overkill reach. Border pieces handle their own neighbours first (fight, else
best capturable neutral, else regroup if the field says a much better frontier exists, else wait or
join a pack). Cap control cancels or detours arrivals past 255 + `CAP_SLACK`.
Lessons from tuning: a fight value that dwarfs neutrals made the whole territory funnel into one
lane of capped pieces and stop expanding (lost 7/10); letting border pieces turn back toward fights
did the same. Results: about 87% vs ImprovedBot in 1v1, 11/12 in 4-player FFA vs 3 ImprovedBots.

## Performance (measured, for the "rewrite in Rust?" question)

50x50, 4 players, 412 turns: 2.7 s total, engine `step` 0.77 s (28%), the rest is bots and pipes.
A faster engine alone saves at most ~30%; parallel games (arena) scale with cores (about 280
games/min on the 10-core Mac with 5 workers). Bots in compiled languages are the bigger lever;
they plug in through `--bot NAME=COMMAND`.

Rust PredatorBot (measured 2026-10-17, machine under load from another arena): 50x50, 4 players,
seeds 1-3 (326, 296, 419 turns): all-Rust 1.88 / 1.67 / 1.79 s, all-Python 4.25 / 3.34 / 4.34 s,
so about 2.2x faster per game. The games are identical (same turns, same death turns) because
the port makes the same moves, which makes this a clean comparison. With Rust bots, waiting on
bots is only ~0.3 s of a game; engine `step` plus serializing and parsing is the rest, so the
engine is now the bottleneck and the next lever if games must go faster.

Rust port fidelity: it reproduces Python's tie-breaking (heap ordered like the tuple
`(-value, site, dir, fight)`, `max()` over `(value, dir, site)` tuples, stable sorts, dict
insertion order for packs and cap control). Checked by running Rust-vs-Python and
Python-vs-Python games on the same seed: replays identical frame for frame. If you change
`PredatorBot.py`, port the change and redo that check. `tournament.py -n 20 -s 11` Rust vs
Python: 11-9.

## Ideas not done yet (only if Maxime asks)

- Starter kits for more languages (Python and Rust exist; the protocol is plain text).
- A stronger reference bot (border-routing of interior strength, overkill tactics,
  non-aggression pact from the late 2016 meta: let others attack first).
- TrueSkill ratings (the arena uses multiplayer Elo).
- Viewer: zoom, per-player highlighting, loading the original 2016 `.hlt` replay format.
- Push to GitHub: no remote is configured. Ask whether public or private before creating one.

## Conventions

- Maxime's writing preference: **never use em-dashes** in prose, docs or messages.
- Keep files self-contained and dependency-free; keep the original starter-kit API names
  even though they are not PEP 8, since compatibility with old bots is the point.
- Commits go on `main`, authored by Maxime Fournes <maxime@pauseia.fr> (the first two commits were made from a sandbox with that identity passed explicitly).
