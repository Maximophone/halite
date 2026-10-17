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
| `arena.py` | Endless tournament: `ProcessPoolExecutor` runs `run_game` in parallel, results appended to `arena/games.jsonl`, multiplayer Elo rebuilt on start from `state.json` (snapshot of all games older than `--history`, default 10000) plus `games.jsonl`; `_compact` folds old games into the snapshot every `COMPACT_SLACK` games and is crash-safe via `last_fold_ids`; stdlib HTTP server for `arena.html`, `/api/state`, `/api/analytics`, `/replays/*.hlt`, pause/resume. Bots: every built `starter-kit/rust/src/bin/*.rs` (re-scanned each game) plus `--bot NAME=CMD`; bots no longer present stay on the leaderboard as `active: false` (shown "retired", below the active ones) |
| `arena.html` | Leaderboard + game list, polls `/api/state` every 3 s; links to `viewer.html?replay=/replays/x.hlt`. Charts (hand-written SVG, no libraries) from `/api/analytics` every 10 s: rating curves, head-to-head matrix, selected-bot panel. Bot colours follow `order` (first appearance), dark palette validated with the dataviz skill against the `#121720` surface |
| `tools/combat_report.py` | Splits each player's strength losses in a replay into damage per opponent, neutrals and the cap |
| `test_rules.py` | Plain-assert rule checks (no pytest needed). Run `python3 test_rules.py` |
| `engine/` | Rust engine, binary `engine/target/release/halite`, std only. `pyrandom.rs` (CPython's MT19937 seeding, `random()`, `_randbelow`, `choice`, `shuffle`), `game.rs` (`generate_map`, `GameState`), `bot.rs` (processes, universal-newline line reader, timeouts), `json.rs` (`json.dumps` byte for byte), `main.rs` (run_game, argparse-like CLI, hidden `--step-test` and `--map-test` modes) |
| `check_engine.py` | Equivalence harness Rust vs Python: maps, single steps, full games, ejection. `python3 check_engine.py [--quick] [--only maps/rules/games/ejection]` |
| `starter-kit/rust/` | The only starter kit (Maxime: "we will only be using Rust now", 2026-10-17). Cargo project, no crates. `src/hlt.rs` is the lib (`hlt`: types plus `get_init`/`send_init`/`Connection::get_frame`/`send_frame`). Bots in `src/bin/`, each a binary in `target/release/`: `my_bot` (Maxime's bot, random template, name `MyBot`), `random_bot` (original sample bot), `improved_bot` (heuristic sparring bot), `predator_bot` (see below), `apex_alpha` (see "Apex line"), `apex_beta` (see "ApexBeta"), `apex_gamma` (strongest, see "ApexGamma"), `hal_lite` (port of a 2016 competitor's bot, see "Bots of 2016 players"). The arena names them by CamelCasing the file stem |

The bots were first written in Python (original 2016 API: `hlt.py`/`networking.py`, camelCase
`getInit`, `getFrame`, ...). ImprovedBot and PredatorBot were ported move for move and checked
identical (40 games each, 2 to 6 players, same seeds and seats, replays equal frame for frame)
before the Python kit was deleted; it is in git history (commit before the Rust-only switch) if
an old Python bot ever needs it. The protocol is unchanged, so 2016-era bots in any language
still run.

Generated, git-ignored: `engine/target/`, `starter-kit/rust/target/` (cargo build output), `arena/` (arena games and replays), `replays/` (`.hlt` + `.html` per game), `logs/bot-N.log` (bot stderr),
`examples/` (a sample game; regenerate with the command in README).

Constraints: **Python 3 standard library only** (tested on 3.10 and 3.13). No packages, no
build step. Keep it that way unless asked. Exception: Rust code (`engine/`, `starter-kit/rust/`),
which Maxime asked for; it uses only the Rust standard library (no crates) and is optional:
everything falls back to Python when it is not built.

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
(cd starter-kit/rust && cargo build --release)            # bots
(cd engine && cargo build --release)                    # engine
python3 halite.py -d "30 30" -s 7 starter-kit/rust/target/release/improved_bot starter-kit/rust/target/release/random_bot
python3 tournament.py -n 6 -s 1 starter-kit/rust/target/release/improved_bot starter-kit/rust/target/release/random_bot
python3 check_engine.py                                 # Rust engine still identical (build it first)
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

## Rust engine (`engine/`, asked for once Rust bots made the Python engine the bottleneck)

Build: `cd engine && cargo build --release` (cargo may be in `~/.cargo/bin`). `halite.py` is the
reference; the Rust engine must play identical games, so **any rule, map, protocol or output change
goes into both**, then run `python3 check_engine.py` (about 90 s; `--quick` about 30 s).

How identity is achieved:
- Maps: `PyRandom` reproduces `random.Random(int)` exactly (init_by_array on the 32-bit words of
  abs(seed), seeds of any size), `random()` (53-bit), `_randbelow` via getrandbits rejection,
  `choice`, `shuffle` (3.10 to 3.13 share the algorithm). The noise is the same float operations in
  the same order (no FMA in Rust by default), `powf` calls the same libm `pow`, `round` is
  `round_ties_even` (Python rounds half to even). Checked on 6,390 maps.
- Rules: `step` is a flat-array port; it keeps the Python quirks (zero-strength pieces on a neutral
  site create a neutral_damage entry and take neutral damage; damage is counted once per target
  slot, so 1- and 2-wide maps double count, as in Python).
- Protocol: Python reads bot stdout in text mode, so the Rust reader splits on universal newlines
  (`\n`, `\r\n`, lone `\r`, waiting for the byte after `\r`), a UTF-8 error ends the stream,
  `strip()` uses Python's whitespace set, `int()` accepts `+7`, `07`, `1_0`. Each bot's 1 s turn
  timeout starts when the engine starts waiting for it, as in Python. A bot whose pipe is closed is
  still read once that turn (and keeps its moves if a line was queued), as in Python.
- Output: same console lines, `-q` JSON with Python's default separators, replay JSON compact with
  ensure_ascii escaping, replay name from local time via libc `localtime_r` (FFI, no crate).
  viewer.html is found by walking up from the binary (the repo root).
- Known difference: Python's `int()` also takes non-ASCII digits (`١`); Rust ignores such a token.

`check_engine.py` covers maps (sizes including 0, negative, 1x1, default random size; seeds 0,
negative, above 2^64), 20,017 random or edge-case steps via `--step-test`, 72 games (1 to 6
players, Rust PredatorBot and ImprovedBot (Python versions until they were retired), `-q`, `--no-html`, `--turns`) and 10 ejection
games with a raw-protocol test bot (crash, timeout, stdout closed, stdin closed with a queued
answer, init crash, init close, missing binary, garbage moves with `\r\n` and a Unicode name).
Not covered: the 15 s init timeout (too slow to test; same code path as the turn timeout).

`halite.play_game(..., engine="auto")` runs the Rust binary with `-q` (or `run_game` when not
built); `tournament.py` and `arena.py` use it and take `--engine auto|python|rust`. Both CLIs got
`--log-dir` and `--no-logs` (the arena runs games with no logs, as before).

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

Rust engine (measured 2026-10-17, load average about 20 from another arena): 50x50, 4 Rust
PredatorBots, seeds 1-3, median of 6: Python engine 2.29 s with .hlt + .html (1.94 s .hlt only,
1.54 s no replay), Rust engine 0.41 s (0.39 s, 0.36 s), so about 5.5x. The Rust engine time is now
mostly the bots themselves. Arena, 5 workers, 3 Rust PredatorBots, 90 s: 252 games/min on the
Python engine, 1,370 games/min on the Rust engine.

Rust port fidelity: it reproduces Python's tie-breaking (heap ordered like the tuple
`(-value, site, dir, fight)`, `max()` over `(value, dir, site)` tuples, stable sorts, dict
insertion order for packs and cap control). Checked by running Rust-vs-Python and
Python-vs-Python games on the same seed: replays identical frame for frame. If you change
`PredatorBot.py` (since deleted; predator_bot.rs is now the only version). `tournament.py -n 20 -s 11` Rust vs
Python was 11-9.

## Apex line (Maxime: "build a bot that crushes the current ones", 2026-10-17)

Before working on bots, read `docs/building-strong-bots.md` (lessons and workflow, written for
the next AI) and use `tools/eval_bot.py` (parallel seeded games vs an opponent pool),
`tools/solo_bench.py` (expansion speed alone) and `tools/trace_game.py` (one game's economy). `apex_alpha.rs` is a frozen prototype; later versions get new names (ApexBeta, ...) so the arena
rates each separately. Every constant reads an `APEX_*` env var (defaults in `Params::load`),
which is how they were tuned. What mattered, measured against PredatorBot:
- Gathering (plan, per frontier target, the turns needed to bring enough strength from up to 8
  steps; rank by value / turns^0.25): +20% production when expanding alone. Ranking by
  value / turns (exponent 1) was 10% worse.
- Free sites (strength 0) must not get value prod/0: it pulled the whole interior toward
  no-man's-land. Value is prod / max(strength, 15).
- Pieces move only from strength 25 (`min_move`): 57% -> 70% 1v1, strength stays in big pieces.
- Defence: pieces within 5 steps of an enemy are not used for expansion, and our sites within 2
  of an enemy pull interior strength: 73% -> 88% 1v1.
- No effect: fight caution (`harm`), fight pull decay, hunting weak players, most expansion
  constants once gathering existed.
Results (100 games, fresh seeds): 89% of 1v1s vs PredatorBot, 53% of 4-player games vs 2
PredatorBots + ImprovedBot, 57% of 6-player games vs 2 PredatorBots, 2 ImprovedBots, RandomBot.

## ApexBeta (2026-10-17)

`apex_beta.rs` = the whole ApexAlpha policy as the default move, then a one-turn combat search for
our pieces within 3 steps of an enemy: an exact simulation of moves, merges, the cap, overkill,
neutral damage and site ownership, scored as enemy strength destroyed - ours lost + production of
sites held; coordinate ascent over each front piece's 5 moves. Enemy moves are predicted by running
our own policy from the enemy's seat (`APEX_PRED=0.6` weight, the rest "enemy stays still").
`APEX_COMBAT=0` plays exactly as ApexAlpha (checked frame for frame).

Why: `tools/combat_report.py` showed ApexAlpha won on economy and only broke even in fights (it
lost as much strength as it dealt vs PredatorBot, even in games it won).

Results (200 games per format, seed 3, fresh): vs PredatorBot pool 95% 1v1, 83.5% ffa4, 75% ffa6
(ApexAlpha: 88 / 61 / 50). vs ApexAlpha 95% 1v1, 56% of ffa4 with 2 ApexAlphas + ImprovedBot.
Sweeps vs itself: prediction is essential (`APEX_PRED=0`: 8.5%), `APEX_LAMBDA=0.7` much worse,
other constants flat.

Fairness rule (Maxime): bots must not rely on knowing opponents' code or identity. ApexBeta never
reads names; it assumes "the enemy plays like me". But its own policy IS ApexAlpha's code, so
results vs ApexAlpha are inflated (exact model). Judge strength on foreign bots (PredatorBot,
ImprovedBot) and in self-play. `APEX_DEPTH=2` (predict enemies as policy + search) beats depth 1
83% but is worse vs foreign bots: level-k exploitation, not strength. Default stays 1.

Prediction accuracy (move vs stay, enemy pieces near us, `APEX_DEBUG=1`): about 0.70 vs
PredatorBot, 0.67 vs ImprovedBot, 0.80 vs ApexAlpha, 0.62 vs ApexBeta. Next step for ApexGamma:
learn the enemy model in game (the `observe` scoring already exists; weight scenarios per
opponent by it), add a greedy-attack scenario, then retune.

## ApexGamma (2026-10-17)

`apex_gamma.rs` = ApexBeta plus a joint search of pairs of front pieces within 2 steps of each
other (all 25 joint moves, after each single-piece pass) and fight value 3 (was 1.5). Fairness:
like Beta, it models enemies with ApexAlpha's policy only (depth 1), so Gamma vs Beta is a fair
test (Gamma does not contain an exact model of Beta).

Results, fresh seeds (`--seed 41`, 1000 1v1 / 400 FFA games vs ApexBeta; `--seed 4`, 200 vs the
PredatorBot pool): 62.8% of 1v1s vs ApexBeta; FFA vs ApexBetas unchanged (4-player 33.8% vs
Beta's own 32.0% on the same seeds, 6-player 31.8% vs 31.2%). PredatorBot pool, Gamma vs
Beta on the same seeds: 1v1 97.5% vs 97%, 4-player 76% vs 74%, 6-player 71% vs 68.5%. Worst
turn 7 ms (6-player 50x50).

Lessons (all at 1000 games; one standard error is 1.6 points):
- Pairs: 58.9% vs Beta, 48.1% with pairs off on the same seeds. Triples: no further gain. Passes,
  margin, prediction weight: flat. The search is saturated; the next gain is in the objective.
- Learned enemy model (per opponent, chance that a piece predicted to move does move: 0.4 vs
  ApexBeta, 0.56-0.68 vs others; pieces predicted to stay move only 3-9% of the time), with 8
  sampled independent scenarios: worse (44%). Correlated "all follow" beats independent sampling.
- Mirror games are decided in the ~60 turns after first contact: production is equal at contact
  (815 vs 803), 60 turns later the winner leads 1791 to 1433 (112/117), combat losses equal,
  winner spent more on neutrals. But pulling strength away from fights (fight value 0.7) is worse
  and more pull (3-6) is better, so this is effect, not cause.
- FFA: weighting enemy losses less (lambda 0.5-0.7) is much worse in every format.
- Solo benchmark: Gamma = Beta = Alpha (no enemies). All expansion knobs flat within 1%. 25% of
  production is lost to moving pieces (a moving piece produces nothing that turn), and 75% of
  strength sits in the interior, two thirds of it in transit: routing is the economy lever left.
- Experiments are the main time cost; the arena competes for CPU, pause it during sweeps.

## Economy routing: tried after ApexGamma, all negative (2026-10-17)

Solo games, turns 40-120: production lost to moving is 18.8% from interior moves, 3.8% from
border captures, 2.5% other; pieces under 60 strength make 41% of interior moves (35% of that
loss); idle border pieces that could capture something hold only 5% of border strength.
Mechanisms tried on the solo benchmark (80 maps, Gamma = 930.7 / 2328 at turns 100 / 150):
- Distance-aware waiting (move threshold grows per step from the border): worse, monotonically
  (-1% to -7% at turn 150). Faster delivery beats saving production.
- Cheaper paths (among near-equal field steps, land on the poorest site): flat.
- Moving earlier (wait 2, min move 8 or 10): flat to -2%.
- Payback ranking of targets, 1 / (w * turns to gather + strength / value): worse at every w
  (-1.7% to -7%). The empirical value / turns^0.25 stays.
Conclusion: routing sits at a broad optimum; the ~19% is the price of fast delivery. The next
economy gain needs planning (e.g. a searched opening), not routing tweaks. The variant file
(apex_delta.rs) was dropped since its defaults equalled ApexGamma.

## Bots of 2016 players (Maxime asked to rebuild top players' bots from their write-ups, 2026-10-17)

Sources found (both on GitHub, code plus write-up):
- David Ma (GitHub `djma`): https://github.com/djma/halite-bot, Java, bot "HAL-lite", written for
  Two Sigma's internal summer 2016 contest, submitted unchanged to Halite I: 19th. The README is
  the write-up (two modes, greedy expansion with a 2-step lookahead, enlisting 255 strength to
  fights, repulsion map, collision resolution).
- Declan Groves (GitHub `DexGroves`): https://github.com/DexGroves/halite-bot, Python + numpy +
  scipy, ranked 7th. Write-up "Halite Bot Breakdown" (dexgroves.github.io/halite/2017/02/15/
  halite.html, now 404; the Web Archive copy has it, formulas are MathJax in the raw HTML).

Neither Java nor scipy is installed here, so the ports were checked by reading, not by running the
originals side by side (both originals also use unseeded randomness, so frame-for-frame identity
is impossible anyway).

`hal_lite.rs`: port of `HAL-lite/MyBot.java` (v15b) with its quirks (BFS visited sets shared
between siblings, child targets computed from the candidate's strength, the never-filled
`usedForExpansion`). The Java bot iterates `HashMap<Location, ...>` where order matters and
`Location.hashCode()` is `x*1000+y`, so `JMap` reproduces Java's iteration order (bucket
`hash & (cap-1)`, insertion order within a bucket, capacity doubling past 0.75 load and kept by
`clear()`; treeified buckets are not modelled). Shuffles use a map-seeded generator. Expansion
budget 800 ms (`HAL_TIME_MS`, Java: 900) never binds: worst turn 10 ms on 50x50 6-player.
Results (100 games per format, `--seed 5`): vs PredatorBot pool 86% 1v1, 83% ffa4, 68% ffa6
(ApexGamma on its own seeds: 97.5 / 76 / 71); vs ApexGamma 34% of 1v1s, 32% of ffa4 (2 ApexGammas
+ ImprovedBot). A 19th-place 2016 bot is a real opponent for the Apex line.

## Ideas not done yet (only if Maxime asks)

- Starter kits for more languages (only Rust now; the protocol is plain text).
- A stronger reference bot (border-routing of interior strength, overkill tactics,
  non-aggression pact from the late 2016 meta: let others attack first).
- TrueSkill ratings (the arena uses multiplayer Elo).
- Viewer: zoom, per-player highlighting, loading the original 2016 `.hlt` replay format.
- Push to GitHub: no remote is configured. Ask whether public or private before creating one.

## Conventions

- Maxime's writing preference: **never use em-dashes** in prose, docs or messages.
- Keep files self-contained and dependency-free (Python stdlib, Rust std, no crates). Never
  change the bot protocol: compatibility with old bots is the point.
- Commits go on `main`, authored by Maxime Fournes <maxime@pauseia.fr> (the first two commits were made from a sandbox with that identity passed explicitly).
