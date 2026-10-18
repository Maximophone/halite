# CodexBot validation

Validated on 18 October 2026. CodexBot won a majority of duels against all 13 other built bots. The executable is installed at `starter-kit/rust/target/release/codex_bot` and the arena discovers it as CodexBot. The same arena entry was reset at the user's request after validation, giving this release a fresh record. These controlled benchmark results are unchanged.

## Final holdout: duels

Seed set 1001 was held out from tuning. Maps range from 20 to 50, with both seats played on each map. Each opponent binary and the candidate were frozen before its evaluation.

| Opponent | Wins / games | Win rate |
|---|---:|---:|
| ErdmanBot | 313 / 400 | 78.25% |
| ApexAlpha | 97 / 100 | 97.00% |
| ApexBeta | 96 / 100 | 96.00% |
| ApexDelta | 89 / 100 | 89.00% |
| ApexGamma | 95 / 100 | 95.00% |
| DexBot | 92 / 100 | 92.00% |
| HAL-lite | 99 / 100 | 99.00% |
| ImprovedBot | 98 / 100 | 98.00% |
| MyBot | 100 / 100 | 100.00% |
| NeuroAlpha | 96 / 100 | 96.00% |
| NeuroBeta | 89 / 100 | 89.00% |
| PredatorBot | 98 / 100 | 98.00% |
| RandomBot | 100 / 100 | 100.00% |

All 1,600 duel games completed with zero candidate or opponent crashes/timeouts.

## Multiplayer

| Opponents | Wins / games | Win rate | Average rank |
|---|---:|---:|---:|
| Three ErdmanBots | 90 / 200 | 45.00% | 2.15 |
| ErdmanBot, DexBot, ApexDelta, HAL-lite, NeuroBeta | 48 / 120 | 40.00% | 2.47 |
| Five ErdmanBots | 39 / 180 | 21.67% | 3.22 |

These multiplayer results use seed set 1001 and all seats on each map. Equal-share first-place rates are 25% for four players and 16.67% for six players. Six-player games against only ErdmanBots remain the hardest tested format; the bot does not win every game.

## Supplementary mixed-field run

Against ErdmanBot, DexBot, ApexDelta, HAL-lite and NeuroBeta in six-player games,
CodexBot finished first in 48 of 120 games (40%). However, NeuroBeta timed out
in nine games. This result is excluded from the headline performance claims.
The 111 games without an ejection contained 45 CodexBot wins; that conditional
subset is exploratory, not a replacement controlled benchmark. CodexBot itself
had no crash or timeout in any of the 2,100 final validation games.

## What changed

- Duel policy: release neutral walls, remove extra strategic waiting, use an interior percentile of 0.75 and a combat threshold of eight times production.
- Multiplayer policy: preserve diplomatic walls and use a 0.4 routing distance penalty; switch to the duel policy when only one rival survives.
- Retain Erdman’s economy, synchronized gathering, overflow checks, and local danger avoidance.
- Handle a fully owned map without an empty-frontier panic, and bound the interior percentile index.

A separate combat search, aggressive multiplayer wall breaking, production exponents, and other parameter combinations were measured and rejected. Development results are preserved in `development_results.json`. The final bot retains attribution to Erdman and the repository’s Rust port.

## Checks and reproducibility

- Three bot regression tests pass, covering a fully owned map, empty/zero-production terrain, and transition from multiplayer to a duel.
- All ten rule checks and seven arena rating checks pass.
- Rust formatting and Python compilation checks pass.
- The installed executable matches the tested binary by SHA-256.

See `README.md` for build and repeat-evaluation commands, and `validation_results.json` for scores and executable hashes. Per-game records and frozen binaries are under `starter-kit/rust/target/codex-results/`.

EvoBot was under construction in the concurrent Claude session and had no built executable at validation time; it is not included in the 13 opponents. Bot development left existing sources and Claude’s changes untouched. The subsequent requested rerelease added persistent reset support to `arena.py`, resetting only CodexBot's record while preserving all other bots' accumulated ratings and totals.

Source SHA-256: `4918b72a3aa4acb56d73860c9edece029a120a6a67646723cc8ce65966752ad1`

Binary SHA-256: `ebdb46e6f24dc9cdf9497d2a8cd80a67958b7645263b95dbc86f8e96b5e38452`
