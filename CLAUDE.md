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
| `arena.py` | Endless tournament: `ProcessPoolExecutor` runs `run_game` in parallel, results appended to `arena/games.jsonl`, skill ratings (Weng-Lin Plackett-Luce, see "Arena ratings") rebuilt on start from `state.json` (snapshot of all games older than `--history`, default 10000, per-bot `[mu, sigma]` under `skill`) plus `games.jsonl`; `_compact` folds old games into the snapshot every `COMPACT_SLACK` games and is crash-safe via `last_fold_ids`; stdlib HTTP server for `arena.html`, `/api/state`, `/api/analytics`, `/replays/*.hlt`, pause/resume. Bots: every built `starter-kit/rust/src/bin/*.rs` (re-scanned each game) plus `--bot NAME=CMD`; bots no longer present stay on the leaderboard as `active: false` (shown "retired", below the active ones) |
| `arena.html` | Leaderboard + game list, polls `/api/state` every 3 s; links to `viewer.html?replay=/replays/x.hlt`. Charts (hand-written SVG, no libraries) from `/api/analytics` every 10 s: rating curves, head-to-head matrix, selected-bot panel. Bot colours follow `order` (first appearance), dark palette validated with the dataviz skill against the `#121720` surface |
| `tools/check_erdman.py` | Plays ErdmanBot and Travis Erdman's original `erdman_v26.py` (path to a clone of his repo, not vendored: no license) on the same seeds and checks the replays are identical |
| `tools/combat_report.py` | Splits each player's strength losses in a replay into damage per opponent, neutrals and the cap |
| `test_rules.py` | Plain-assert rule checks (no pytest needed). Run `python3 test_rules.py` |
| `test_ratings.py` | Plain-assert checks of the arena ratings and the snapshot rebuild. Run `python3 test_ratings.py` |
| `engine/` | Rust engine, binary `engine/target/release/halite`, std only. `pyrandom.rs` (CPython's MT19937 seeding, `random()`, `_randbelow`, `choice`, `shuffle`), `game.rs` (`generate_map`, `GameState`), `bot.rs` (processes, universal-newline line reader, timeouts), `json.rs` (`json.dumps` byte for byte), `main.rs` (run_game, argparse-like CLI, hidden `--step-test` and `--map-test` modes) |
| `check_engine.py` | Equivalence harness Rust vs Python: maps, single steps, full games, ejection. `python3 check_engine.py [--quick] [--only maps/rules/games/ejection]` |
| `starter-kit/rust/` | The only starter kit (Maxime: "we will only be using Rust now", 2026-10-17). Cargo project, no crates. `src/hlt.rs` is the lib (`hlt`: types plus `get_init`/`send_init`/`Connection::get_frame`/`send_frame`). Bots in `src/bin/`, each a binary in `target/release/`: `my_bot` (Maxime's bot, random template, name `MyBot`), `random_bot` (original sample bot), `improved_bot` (heuristic sparring bot), `predator_bot` (see below), `apex_alpha` (see "Apex line"), `apex_beta` (see "ApexBeta"), `apex_gamma` (see "ApexGamma"), `apex_delta` (Gamma plus DexBot's tunnel value, see "ApexDelta"), `hal_lite`, `dex_bot` and `erdman_bot` (strongest, the 2016 winner; ports of 2016 competitors' bots, see "Bots of 2016 players"), `neuro_alpha` (convolutional network cloned from DexBot, see "Neural line"). `src/nn.rs` (module `hlt::nn`) holds the network's features and std-only forward pass; `weights/neuro_alpha.hnn` is compiled into the bot. The arena names them by CamelCasing the file stem |
| `rl/` | Neural training, the one place with crates (Maxime chose candle on Metal, 2026-10-17): `src/main.rs` (`halite-rl train`, `parity`, `bench`), `src/metal_conv.rs` (hand-written Metal kernels for the wrap-around conv), `gen_data.py` (teacher games), `dashboard.py` + `dashboard.html` (live training curves, port 8090). See "Neural line" |

The bots were first written in Python (original 2016 API: `hlt.py`/`networking.py`, camelCase
`getInit`, `getFrame`, ...). ImprovedBot and PredatorBot were ported move for move and checked
identical (40 games each, 2 to 6 players, same seeds and seats, replays equal frame for frame)
before the Python kit was deleted; it is in git history (commit before the Rust-only switch) if
an old Python bot ever needs it. The protocol is unchanged, so 2016-era bots in any language
still run.

Generated, git-ignored: `engine/target/`, `starter-kit/rust/target/` (cargo build output), `arena/` (arena games and replays), `replays/` (`.hlt` + `.html` per game), `logs/bot-N.log` (bot stderr),
`rl/target/`, `rl/data/` (teacher replays), `rl/runs/` (training logs and checkpoints), `rl/checkpoints/` (snapshots and their evaluations),
`examples/` (a sample game; regenerate with the command in README).

Constraints: **Python 3 standard library only** (tested on 3.10 and 3.13). No packages, no
build step. Keep it that way unless asked. Exception: Rust code (`engine/`, `starter-kit/rust/`),
which Maxime asked for; it uses only the Rust standard library (no crates) and is optional:
everything falls back to Python when it is not built. Second exception: the training crate `rl/`
(candle, Metal), which nothing else depends on; the bots it trains stay std-only.

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
python3 test_ratings.py                                 # arena rating checks
(cd starter-kit/rust && cargo build --release)            # bots
(cd engine && cargo build --release)                    # engine
(cd rl && cargo build --release && ./target/release/halite-rl bench)   # Metal kernels vs reference, step timing
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
(an earlier apex_delta.rs) was dropped since its defaults equalled ApexGamma; the name now
belongs to the tunnel bot (see "ApexDelta").

## Combat look-ahead: tried after ApexGamma, negative (2026-10-17)

Added to the combat score a heuristic of next turn: after the simulated turn, each surviving
piece is exposed to the enemy strength within 2 steps (min of the two counts as lost), the enemy's
pieces likewise to ours, and a site within reach of more strength than it holds counts as lost.
Weight 0 reproduced ApexGamma frame for frame; worst turn 33 ms. vs ApexGamma (400 1v1 games,
seed 51; control 52.0%): weight 0.1 47.0%, 0.25 45.0%, 0.5 44.5%, 1 37.5%; without the site term
41.5%; site term x3 46.8%. Offence only (credit for the enemy's exposure, no penalty for ours):
13.0% at 0.25, 4.2% at 1. Our-exposure penalty makes the bot passive, the credit alone makes it
reckless; the exact one-turn score beats both. Looking further needs a real two-turn simulation
(both sides re-planned), not a proxy. The variant file was dropped.

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

`dex_bot.rs`: port of DexBot at HEAD of his repo (his README: the submitted bot is 780144e plus
the non-aggression changes of HEAD; the rest of the diff is comments and last-turn handling).
Pipeline per turn: Combatant (fights at our border, pieces within 8 steps walk to the nearest
fight), MoveMaker (border value `G = prod^2/strength` blurred by a Gaussian sigma 1.2, plus a tunnel
value `T_i = sum_j G_j / D_ij` over the neutrals j closest to border site i, D an all-pairs
Dijkstra computed once at init; idle pieces strongest first take `argmax G/(dist+t) + k T/(dist+10)`
with `k = 1.5 - 0.1 * enemies`), Resolver (chessboard: only `(x+y+turn)` even sites follow the plan
after turn 40 or once in combat, odd ones dodge; cap control), Nonaggressor (never break a neutral
wall touching an enemy, except the weakest once we are twice as strong). numpy semantics kept:
`[x, y]` arrays (`k = x*h + y`), first-index argmax, integer truncation of `combat_heur /= 10000`,
scipy's sparse Dijkstra dropping zero-cost edges (`int(strength/(prod+0.1)) == 0`) and taking
either direction (`directed=False`), `safe_to_take` reaching 2. Changes: stable argsort, site-order
sets, seeded `random.choice`, and `np.min([])` (a crash in the original) skips the step (never
fired in 6 instrumented games). Init (2,500 Dijkstras on 50x50) well under the 15 s limit, worst
turn 20 ms.
Results (100 games per format, `--seed 5`): vs PredatorBot pool 100% 1v1, 95% ffa4, 85% ffa6;
vs ApexGamma 65% 1v1, 51% ffa4, 44% ffa6; vs HAL-lite 74% 1v1, 59% ffa4, 43% ffa6. DexBot is the
strongest bot in the repo. The Apex line's gap is likely the non-aggression pact and the tunnel
value (expansion toward distant rich land), both absent from Apex.

`erdman_bot.rs` (2026-10-18, Maxime: "the bot who won the competition"): Travis Erdman (`erdman`),
the Halite I winner (from memory and his own README; the official final leaderboard could not be
fetched, the Web Archive was offline). https://github.com/erdman/erdman-halite-bots has v12, v17 and
his final v26 with a long README write-up; the port is v26. Pure stdlib Python 3 (his `hlt.py`
became the official Python kit), so the original runs unchanged on this engine (`python3
erdman_v26.py`, about 1 s per turn at worst on 50x50, no timeouts seen). Mechanisms: one potential
field from a Dijkstra-style search seeded with strength / production at every unowned square
(empties next to enemies -0.5 per enemy, walls 100x), each neutral step averaging
`0.9 * potential + 0.1 * s/p` (the tunnelling), each step into our land adding
`0.2 * distance^2`; red-green trees (each owned square points at its best neighbour; per border
target, walk the tree level by level to red-light the squares that must wait and green-light the
ones that must go); moves strongest first with a `destinations` map against the cap; "dangerous
empties" (enemy-adjacent empties one of our moves already touches) are dodged; strategic stilling;
an interior strength hurdle at a percentile that falls as the territory grows; walls touching an
enemy or an empty are not broken while mining remains (his NAP). Kept bug: `seen_enemies` only
ever holds 0 and our id (he describes it in the README). Every tie in his code is broken by
`random.random()`, so the port reimplements CPython's MT19937 (seeding and 53-bit floats, as
`engine/pyrandom.rs`) and draws in the same order (frontier squares row-major, every heap push,
4 draws per assigned square and 5 more in the dodge branch). With `ERDMAN_SEED=S` against the
original after `random.seed(S)`: 14 of 14 games identical frame for frame (1 to 5 opponents, 20x20
to 50x50, `tools/check_erdman.py`); a 15th differed only because RandomBot is clock-seeded.
Results (100 games per format, `--seed 5`; 1v1 / ffa4 / ffa6): vs DexBot 87 / 76 / 72%, vs
ApexGamma 90 / 76 / 68%, vs HAL-lite 97 / 68 / 64%, vs PredatorBot pool 98 / 93 / 86%, vs
ApexDelta 85 / 64 (1v1 / ffa4), vs NeuroAlpha 85 / 73. The winner is in a class of its own here.

## ApexDelta (2026-10-17)

`apex_delta.rs` = ApexGamma plus DexBot's tunnel value (ablating it in DexBot took it from 68% to
35.5% of 1v1s vs ApexGamma). For each frontier site i, T_i sums (prod_j^2 / s_j) / (D_ij + s_j /
prod_j + 1) over the positive-strength neutrals j whose closest frontier site is i, D an all-pairs
Dijkstra on the init map (DexBot's `all_pairs`, computed before the name reply: about 2.5 s on
50x50 under heavy load, inside the 15 s init; computed after it, it timed out turn 1 on big maps).
Knobs: `APEX_TUNNEL_FIELD` (default 1) adds w * k * T / 10 to each frontier target's value, which
seeds the routing field and the gathering ranking; `APEX_TUNNEL` (default 0) adds w * k * T /
(turns + 10) to the gathering score only; `APEX_CHESS` (default 0) is DexBot's chessboard outside
the combat search (after that turn or once in contact, only x + y + turn even sites move; the
others stay unless arrivals would overflow their site). k = 1.5 - 0.1 * enemies. All three at 0 =
ApexGamma frame for frame (5 games, 2 to 6 players). The tunnel is computed only for our own seat:
enemy predictions skip it (gathering never uses pieces within 5 of an enemy, so the predicted
moves near us are unchanged).

Scale: tval (prod / max(s, 15) + ...) is about 0.05-0.2 for neutrals, k * T up to about 1, so
weight 1 makes the tunnel comparable with the local value on the best frontier sites.

Solo benchmark (80 maps, turns 100 / 150; Gamma 930.7 / 2328.1): `APEX_TUNNEL` 0.25: 1010 / 2463,
0.5: 1031 / 2497, 1: 1036 / 2524, 2: 1027 / 2513, 4: 1011 / 2491, 8: 989 / 2468;
`APEX_TUNNEL_FIELD` 0.5: 1029 / 2490, 1: 1045 / 2524, 3: 1018 / 2497. Both together: no further
gain (1028 / 2524). Chessboard (turn 40) costs 3% (1004 / 2467 on top of `APEX_TUNNEL=1`).

Win rates, 200 games per format, `tools/eval_bot.py --suite 1v1,ffa4` (ffa4 = candidate, 2
opponents, ImprovedBot). Machine at load 60-100 from other sessions throughout; 0 errors in every
line below.

| candidate | vs Gamma 1v1 / ffa4 | vs DexBot 1v1 / ffa4 |
|---|---|---|
| seed 12: ApexGamma | 49 / 37.5 | 36 / 16.5 |
| seed 12: `APEX_TUNNEL=1` | 70 / 53 | 43.5 / 20 |
| seed 12: `APEX_TUNNEL=0.5` / `2` | | 45 / 24, 47 / 25.5 |
| seed 12: `APEX_TUNNEL_FIELD=1` (default) | 70.5 / 56 | 45 / 20.5 |
| seed 12: `APEX_TUNNEL_FIELD=0.5` / `2` | | 49 / 23.5, 43.5 / 23 |
| seed 12: both at 1 | 67.5 / 55 | 45.5 / 25 |
| seed 12: default + `APEX_CHESS=40` | 75 / 58.5 | 47 / 23 |
| seed 13 (fresh): ApexGamma | | 27.5 / 13.5 |
| seed 13: default | 76.5 / 46.5 | 42.5 / 17.5 |
| seed 13: default + `APEX_CHESS=40` | 77 / 46 | 44.5 / 16 |
| seed 13: chess only (`APEX_TUNNEL_FIELD=0 APEX_CHESS=40`) | | 33 / 16.5 |

Verdict:
- The tunnel value is a large, real gain: +8% production at turn 150 alone, about 3 in 4 1v1s vs
  ApexGamma, and vs DexBot 1v1 from 27.5-36% to 42.5-45%. All weights from 0.5 to 2 and both
  placements (gathering score or target value) are within noise of each other.
- The chessboard is flat (+2 points 1v1, -0.5 to -1.5 ffa4 on fresh seeds) and costs 3% economy,
  so it is off. DexBot's chessboard gain probably lives in its own crude combat, which Apex's
  search already does better.
- DexBot still leads (about 56% of 1v1s vs ApexDelta) and ffa4 vs 2 DexBots barely moved (13.5 to
  17.5% on fresh seeds). Next candidates: DexBot's non-aggression (flat in DexBot's own ablation,
  but Apex breaks walls differently), and why ffa4 lags (trace a lost seed with
  `tools/trace_game.py`).

## Neural line (Maxime: "start working on a reinforcement learning bot", 2026-10-17)

Maxime finished top 20 in 2016 with a CNN trained on replays (no RL) and asked whether this could
be done in Rust and whether transformers would be better now. Decisions: training in a separate
crate `rl/` with candle on Metal (his choice among pure-std Rust, candle/burn, PyTorch), the bot
std-only; behaviour cloning first, then RL (PPO self-play, not started yet).
`docs/neuro-alpha-infographic.html` explains the architecture for humans.

Architecture (`starter-kit/rust/src/nn.rs`, shared by trainer and bot via `#[path]`, so features
and inference cannot drift): fully convolutional, channel-last, every 3x3 conv wraps around the
map edges. Stem conv 11 -> 32 channels, 10 residual blocks
`h = relu(h + conv3x3_d(h) + b + mean_xy(h) . G)` with dilations 1,1,2,1,2,4,1,2,4,8 (receptive
field 27 steps each way, more than a 50x50 map; the global mean adds map-wide context from block
1), head 1x1 -> 5 logits per site, each owned site plays its argmax. 106,085 parameters. 11 input
planes from the player's seat (mine, enemy, neutral, strength, production, the three
owner x strength products, free neutral site, turn / turn limit, enemies alive / 5).
Why not a transformer: actions are local, the torus is free with circular padding, any map size,
and attention over 2,500 sites is quadratic where a conv layer is linear (about 110 ms per turn
on 50x50 in plain Rust on a loaded CPU, well inside the 1 s limit).

Weights: format "HNN1" (see nn.rs), written as `model.hnn` / `best.hnn` (lowest validation loss)
into the run directory at every evaluation. `neuro_alpha` embeds `weights/neuro_alpha.hnn` with
`include_bytes!` so each binary is a frozen version (new weights for the arena = new bot name);
`NEURO_WEIGHTS=path` overrides them for evaluation, `NEURO_TEMP=t` samples instead of argmax.
`halite-rl parity --run DIR` checks candle against nn.rs: max logit difference 1.4e-6, argmax
identical on 7,678/7,678 sites.

Data and training (behaviour cloning on DexBot):
- `rl/gen_data.py --games 1200 --seed 1` (and `--games 100 --seed 2 --out rl/data/val`): 2 to 6
  players, each seat DexBot with probability 0.6, else ApexGamma, HAL-lite, ApexBeta or
  PredatorBot. About 5 minutes. The trainer imitates the seats named DexBot only: 747,378 frames.
- `halite-rl train --data rl/data/train --val rl/data/val --run rl/runs/NAME --steps 20000`:
  batches of about 65,536 sites drawn from one map size (bucket chosen in proportion to its
  sites), one of the 8 board symmetries at random (move directions remapped), AdamW 1e-3, warmup
  then cosine. Cross-entropy on our sites only, **each frame weighted equally** (1 / sites
  owned): with per-site weighting the first network never moved its starting piece, because a
  one-site opening counted 500 times less than a mid-game frame.
- `python3 rl/dashboard.py` shows any run live (loss, accuracy, move recall, learning rate).
- Validation of run bc3: loss 0.350 / accuracy 88.8% / move recall 49% at step 1,000, 0.238 /
  91.3% / 62% at step 11,000, 0.217 / 91.9% / 66% at step 20,000 (the released weights). Accuracy counts all our sites (always STILL scores 82%); move
  recall is the share of DexBot's non-STILL moves predicted exactly.

Results (`tools/eval_bot.py --seed 7`, 50 games per format, 0 errors), released weights (bc3
step 20,000) with the step-10,000 checkpoint in brackets: vs PredatorBot pool 94% 1v1 (86),
82% ffa4 (70), 64% ffa6 (60), about ApexBeta's level (95 / 83.5 / 75); vs DexBot 14% 1v1 (16),
22% ffa4 (10); vs ApexGamma 44% 1v1 (32). Training longer kept paying off. Cloning reaches a
strong bot but stays well below its teacher: errors compound in positions the teacher never
reaches. That gap is RL's job.

Training speed (the GPU part; data generation is not a bottleneck):
- candle's own conv2d backward on Metal was about 110 ms per layer; a roll-and-concatenate conv
  40 steps/min (candle's narrow backward pads every slice to full size); one `index_select`
  im2col + matmul 72 steps/min (candle's gather moves about 4 GB/s).
- `metal_conv.rs`: hand-written kernels compiled at runtime (no Xcode needed). Scalar conv and a
  tiled weight-gradient kernel: 254 steps/min. Run bc3 trained steps 0 to 13,000 with these.
- Profiling then showed 70% of a step was not the convolution but about 30 small full-size candle
  ops per block (bias, residual, broadcast, relu and their gradients). `FusedConv` does a whole
  layer in one kernel: relu(conv(x) + per-image bias [+ x]); backward = one relu-mask pass, one
  transposed conv with residual and the global-mean gradient folded in as a per-image bias, the
  weight gradient, one sum. The global mean is detached in the graph for that reason.
- Matrix units: `conv_fwd_sg` and `conv_wgrad_sg` use `simdgroup_matrix` 8x8 blocks, 32-pixel
  tiles staged in threadgroup memory. They only helped once staging lost its per-element integer
  division and modulo (neighbour addresses computed once per tile, Cin padded to a power of two).
  On these sizes their sums come out bit-identical to the scalar kernels.
- Full step, 26 x 50x50 batch, measured while bc3 shared the GPU: unfused scalar 393 ms, fused
  scalar 240 ms, fused matrix 132 ms (3x). In real training: bc3 resumed from step 13,000
  (`--resume 13000`, Adam state restarts, no visible bump) at 734 steps/min against 251 (2.9x). `bench` checks every gradient of the whole model against
  the plain candle reference (relative difference under 1e-6) and prints these timings.
- `--conv fused|metal|gather` picks the path (`fused` is the default and uses the matrix kernels
  when Cout is 8, 16, 32 or 64 and Cin <= 64).

### RL (PPO self-play, started 2026-10-18)

`halite-rl rl --init rl/runs/bc3/model.safetensors --run rl/runs/ppo1 --lr 5e-5` (`src/ppo.rs`,
every knob documented at its top):
- Simulator `src/sim.rs`: the engine's own `game.rs` and `pyrandom.rs` included at the crate root
  plus the engine's game loop (deaths, end, turn limit, ranking). `halite-rl simtest --dir
  rl/data/val --games 100`: 100/100 engine replays identical (map from the seed, 28,174 frames,
  end turn, ranking).
- Rollout: 48 games per iteration in 8 (map side, players) types, so each turn is a few batched
  forward passes. One seat per game is the learner, other seats are the learner with probability
  0.5, else a frozen opponent playing argmax (the starting network, then a snapshot every 25
  iterations, at most 3 kept). Learner moves sampled from softmax(logits / 0.5), the same
  temperature in the update; at temperature 1 the sampled policy was far weaker than argmax
  (5% wins per seat against 35%).
- Reward: final rank (+1 first .. -1 last) at the seat's last turn, plus the change of its share
  of player-owned production each turn (a potential). GAE (gamma 0.999, lambda 0.95), 25% of
  learner frames kept for the update, PPO clip 0.2, per-site ratio, each frame weighted equally,
  entropy 0.003, 2 epochs.
- Value head: MLP (64) on the *detached* trunk means plus map means of the input planes and of
  owner x production. With the value loss flowing into the shared trunk, the value-only warmup
  wrecked the cloned policy (entropy 0.29 -> 0.60, learner wins to 0); detached, warmup leaves the
  policy exactly unchanged (KL 0) and explains about 45-55% of returns from the start.
- Speed, first version: about 46 s per iteration for 48 games (rollout 36 s, update 9 s), the
  rollout GPU bound (`sample` showed 90% waiting on results). Pipelining the per-group forward
  passes and doubling the games per call changed nothing at that point. What worked:
  - Occupancy, not tile size: 128-pixel tiles (24 KB of threadgroup memory) were 2x *slower*;
    shrinking the 32-pixel kernels' threadgroup memory (forward 25 KB -> 7 KB, weight gradient
    16 KB -> 8 KB, layers limited to 32 channels) made them 2.2x and 3x faster: several
    threadgroups per core hide the latency of the scattered neighbour reads.
  - `metal_conv::pixel_mean`: candle's mean over a middle dimension ran at about 8 GB/s; the
    trunk takes one per block and the value head six. A two-stage kernel is 2x faster and the
    value head now does one pass for all its means.
  - Result, steady state with nothing else on the GPU: rollout 2.1x faster (about 800
    game-turns/s against 380), an iteration of 48 games 29 s against 45 s. Remember that the
    first iteration of any run includes compiling the GPU kernels: never time it.
- Run ppo1 (48 games, lr 5e-5, no teacher KL): the policy drifted the wrong way. Learner rank
  score per seat fell from 0.459 to 0.410 over 19 iterations while the frozen BC net's rose
  from 0.553 to 0.612 (KL per iteration only about 0.005, entropy stable). Against real bots at
  iteration 25: 87.5% 1v1 vs the pool (unchanged), 67.5% ffa4 (75%), 12.5% 1v1 vs DexBot (25%).
  Diagnosis: one advantage per frame spread over hundreds of sites is mostly noise, and a noisy
  walk away from a cloned optimum makes it worse. Fix in ppo2: `--teacher-kl 0.1`
  (KL(policy || starting network) per site, what AlphaStar and the Lux AI winners did) and 96
  games per iteration.
- Run ppo2 (96 games, teacher KL 0.1, 300 iterations, about 5 h overnight): stable but flat. Every
  25th checkpoint against real bots (100 games per format): 88-94% 1v1 vs the pool, 70-77% ffa4,
  18-27% 1v1 vs DexBot, all within noise of the start (94 / 71 / 24). Teacher KL stayed about
  0.017 nats per site; the policy never left the cloned network's neighbourhood.
- Gradient signal (`--snr 1`: policy gradient on even vs odd games, cosine). Per-frame advantage,
  from ppo2's weights: -0.05 (48 games per half), 0.17 and -0.05 (96, 25% / all frames), 0.32
  (96, another batch). The estimate itself is noisy; the signal is weak, a usable gradient needs a
  few hundred games per update. Local credit (per-site return over a 9x9 window, horizon about 10
  turns, value-map baseline, `--credit local`) measured 0.32 with the global term and 0.11 alone,
  after only 3 warmup iterations of the value map (its loss still falling fast): not better yet.
- Where NeuroAlpha loses to DexBot (`rl/analyse_matchup.py`, 40 games, 9 won): in lost games it
  is 13% behind in production at first contact (turn 51: 847 vs 979), then stalls while DexBot
  keeps expanding (60 turns later 937 vs 1,418), and it loses 18% more strength in fights than
  it destroys. In won games contact comes later (turn 92) and it is ahead. Solo benchmark
  (production at turns 60 / 100 / 150): NeuroAlpha 217 / 944 / 2,463, DexBot 233 / 994 / 2,587,
  ApexGamma 243 / 980 / 2,565. So the economy gap is 5-7%, the bigger one opens after contact.
- `--solo P` makes a share of the games single-player (cut at `--solo-turns`, reward = share of the
  map's production): Maxime's idea of training the economy alone. Cheap (one forward pass per
  turn) and noise-free on the opponent side: the cleanest test of whether the algorithm can
  improve anything. `eval_checkpoints.py --solo` scores checkpoints on the solo benchmark.
- Run solo1 (from bc4, 192 solo games per iteration cut at turn 150, frame credit, teacher KL
  0.02, 50% of frames): the solo benchmark at turn 150 fell 2,562 -> 2,478 (iter 10) -> 2,439
  (iter 20). The in-training production share swung 0.40-0.65 between iterations from the map
  draws alone, so never judge a run on it. Third PPO run to get worse or stay flat: the updates
  are mostly noise, and a noisy step from a trained policy is usually a step down.

### bc4 and DAgger (2026-10-18)

- bc4: bc3's weights fine-tuned on 3,200 teacher games (`rl/data/train`, `rl/data/train2`, 2M
  frames), 40k steps, lr 5e-4. Validation loss 0.194 (bc3 0.217), move recall 69% (66%), solo
  benchmark 2,562 at turn 150 (2,463; DexBot 2,587). Same seeds (23), 100 games each, bc4 vs the
  released bc3: 1v1 pool 95 vs 94, ffa4 72 vs 71, ffa6 74 vs 77, 1v1 DexBot 26 vs 24, 1v1
  ApexGamma 33 vs 34. Better imitation no longer turns into wins: the cloning plateau (errors
  compound in positions the teacher never reached). Not released.
- DAgger: `rl/target/release/dagger` (rl/src/bin/dagger.rs, std only, outside the starter kit so
  the arena ignores it) plays the network's argmax while a shadow DexBot process gets the same
  frames; DexBot's answers for our sites are written per turn to a `.dag` file (format in the
  file's doc). `rl/gen_dagger.py --weights W --games 1500` plays one DAgger seat per game against
  DexBot (half the seats) and the other strong bots: 1,500 games in about 25 minutes, 1.8 GB. In
  those positions DexBot moves 26% of our sites (18% in its own games). The trainer reads `.dag`
  files and weights directories with `dir*k`.
- dag1 (bc4 + teacher data + `dagger1*3`, 20k steps, lr 3e-4): clearly worse. Same seeds, 100
  games: 1v1 pool 80 (bc4 95), ffa4 63 (72), 1v1 DexBot 10 (26); validation loss on DexBot's own
  games 0.213 (0.194). The mid-run checkpoint (step 8,000) already had 12% vs DexBot.
- The labels are right: with `DAGGER_BETA=1` the wrapper plays the shadow's recorded answers and
  wins 52.5% of 40 1v1s vs DexBot (DexBot vs itself). DexBot recomputes everything from the frame
  (its only cross-turn state is the sticky `in_combat`, the turn and an RNG), so its answers in
  our positions mean what they should. Untested explanations: 3x weight on mostly losing
  positions (we lose 3 in 4 of these games) teaches desperate play that leaks into normal
  positions the features cannot tell apart; DexBot's chessboard (x + y + turn parity) makes its
  per-site moves look random to a network that only sees turn / limit; capacity (106k params)
  shared between two distributions. Next to try if DAgger is resumed: weight 1, beta 0.5
  (expert plays half the turns, so states stay near its own), or only the turns before contact.

### Capacity, features and ErdmanBot as teacher (2026-10-18)

- Capacity: from scratch on the 3,200 DexBot games, same seed, validation loss at steps 1k / 2k /
  3k: 32 channels x 10 blocks (106k params) 0.368 / 0.332 / 0.303, 64 x 12 (about 490k params)
  0.344 / 0.296 / 0.279. The big net imitates better at every step (small net's 10k: 0.241). The
  64-channel kernels (`*_sg64`, MAXC 64) are correct but less efficient: 112 steps/min against
  860. nn.rs threads the convolutions for nets over 48 channels (`NEURO_THREADS`); the big net
  takes about 7 ms per 30x30 turn on the CPU.
- Feature set v2 (`--features 2`, 24 planes, see `features_v2` in nn.rs): DexBot's parity,
  distances (enemy, depth in our land, distance to it), capture value, an ErdmanBot-style
  potential field, my shares of production / sites / strength, per-enemy relative strength,
  walls, own neighbours. About 0.1 ms per frame. Weights record their plane count (`arch[0]`)
  and `Net` picks the feature set, so v1 weights (NeuroAlpha) still work. `halite-rl features`
  prints the planes of a few frames.
- ErdmanBot as teacher: `gen_data.py --teacher erdman_bot`, 4,000 games (`rl/data/erdman`, 9.6
  GB) + 150 validation. Small net, 10k steps, v1 vs v2: validation loss 0.3803 vs 0.3800 (no
  difference; DexBot cloning reaches 0.24 at the same point: Erdman breaks every tie with
  `random.random()`, so many of its moves are unpredictable). In games (100 each, seed 23): vs
  DexBot 1v1 4% / 5%, PredatorBot pool ffa4 21% / 29%, vs ErdmanBot 0% / 2%. The DexBot clone
  at the same step count had 70% ffa4. Cloning the winner works far worse than cloning DexBot:
  ErdmanBot's strength is precise coordination (red-green trees, which pieces wait), which an
  approximate copy loses, while DexBot's rules degrade gracefully. v2 features: no imitation
  gain, maybe a small one in play (+8 points ffa4, about 1.7 standard errors).
- Monitoring: the dashboard shows RL runs (rank score and win rate per seat of learner vs frozen
  opponents, explained variance, entropy). `python3 rl/eval_checkpoints.py rl/runs/ppo1 --every 25`
  plays every 25th checkpoint (argmax, i.e. the real bot) against the PredatorBot pool and DexBot,
  same seeds each time, into `rl/runs/ppo1/evals.txt`.

## Arena ratings (2026-10-17, replaced multiplayer Elo)

Maxime asked whether TrueSkill would beat the pairwise multiplayer Elo. Not strictly, but a
rating with uncertainty ranks new bots honestly sooner and rates an N-player finishing order
natively. Chosen: the Weng-Lin Bayesian approximation with the Plackett-Luce model (Weng and Lin,
JMLR 2011, what the `openskill` package implements), not TrueSkill itself: same idea (a mean mu
and an uncertainty sigma per bot), patent-free, and about 40 lines (`rate` in `arena.py`) with no
Gaussian message passing. Ties (equal ranks) are supported, though the engine never produces them.

- Parameters: `MU = 25`, `SIGMA = MU / 3`, `BETA = MU / 6`, `KAPPA = 1e-4` (openskill defaults),
  `TAU = MU / 3000` (openskill uses MU / 300). Replaying the 10,400 listed games: with MU / 300
  sigma settled near 1.7 and the conservative curves jittered by about 1.3; with MU / 3000 sigma
  settles near 0.7 and the jitter halves, while the order of the bots is the same. TAU 0 differs
  little over 10k games but would eventually freeze ratings, which matters if `my_bot` changes.
- Leaderboard order and the rating curves use the conservative rating mu - 3 sigma; the table also
  shows mu and sigma. A new bot starts at 0 conservative and needs about 50 games to halve sigma.
- A bot with several seats in one game (fewer bots than seats) gets the sum of its seats' mean
  changes and the product of their variance shrinks.
- Migration: an old `state.json` holds Elo under `ratings`, which cannot be converted. On load
  it is dropped and skills restart from the games still in `games.jsonl` (10,000 is plenty for
  the estimates to settle); win and game totals are kept. First run on Maxime's data: ApexBeta
  37.8, ApexGamma 37.3 (only 3k games, still close), ApexAlpha 28.2, PredatorBot 23.5,
  ImprovedBot 18.9, RandomBot 13.9, MyBot 13.1.

## Ideas not done yet (only if Maxime asks)

- Starter kits for more languages (only Rust now; the protocol is plain text).
- A stronger reference bot (border-routing of interior strength, overkill tactics,
  non-aggression pact from the late 2016 meta: let others attack first).
- Viewer: zoom, per-player highlighting, loading the original 2016 `.hlt` replay format.
- Push to GitHub: no remote is configured. Ask whether public or private before creating one.

## Conventions

- Maxime's writing preference: **never use em-dashes** in prose, docs or messages.
- Keep files self-contained and dependency-free (Python stdlib, Rust std, no crates). Never
  change the bot protocol: compatibility with old bots is the point.
- Commits go on `main`, authored by Maxime Fournes <maxime@pauseia.fr> (the first two commits were made from a sandbox with that identity passed explicitly).
