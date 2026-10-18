# CodexBot experiments

CodexBot is an independent challenger derived from the existing ErdmanBot port.
It retains Erdman's economy and gathering algorithm, with separate policies for
duels and games with several opponents. It also handles a fully owned map without
trying to search an empty frontier.

In duels it releases neutral walls, removes the extra strategic-waiting rule,
and uses a 0.75 interior percentile with a combat threshold of eight times site
production. In multiplayer it preserves the diplomatic walls and uses a 0.4
routing distance penalty, directing reserves more locally. Once only one rival
remains, it switches to the duel policy.

This work adds `starter-kit/rust/src/bin/codex_bot.rs` and this experiment directory.
It does not require changes to the engine, existing bots, shared documentation,
or Claude's work. Experimental builds use their own Cargo target directories.

## Build

```sh
cargo build --release --manifest-path starter-kit/rust/Cargo.toml \
  --target-dir starter-kit/rust/target/codex-final --bin codex_bot
```

The normal `cargo build --release --bin codex_bot` also works. The arena discovers
the bot when its executable is present in `starter-kit/rust/target/release/`.
The release updates that same CodexBot entry. On 18 October 2026, at the user's
request, its early arena record was reset for a fresh release. The arena records
an ordered reset event so old results still reconstruct opponents' ratings
exactly, while CodexBot starts with zero games and its previous release is hidden
from match history and analytics. A reset requires pausing and draining matches;
`Arena.reset_bot("CodexBot")` persists through restarts and compaction. Backups
are under `arena/backups/`. Controlled benchmark results here are independent
of the arena record. `test_arena_reset.py` verifies this behavior.

## Reproduce an evaluation

```sh
python3 tools/codex_experiments/sweep.py tools/codex_experiments/release.json \
  --candidate starter-kit/rust/target/codex-final/release/codex_bot \
  --opponents erdman_bot --games 200 --seed 1001 --workers 4 \
  --out starter-kit/rust/target/codex-results/recheck.json
```

Use `--players 4` or `--players 6` for multiplayer. All opponents are ErdmanBots
in the command above. A comma-separated opponent list produces mixed lineups.
Repeat with each opponent individually to measure head-to-head win rates.
`python3 tools/codex_experiments/validate.py` automates this for every built bot,
using 400 games against ErdmanBot and 100 against each other opponent.

The runner uses map sizes 20 through 50, fixed random seeds, and every seat on
each map. It records every result, both sides' errors, executable hashes, and
immutable binary snapshots. Snapshots prevent concurrent rebuilds from changing
the policy halfway through an evaluation. All output belongs under the ignored
Cargo target directory. RandomBot and MyBot use their own randomness, so their
results need not reproduce exactly even with the same maps.

## Configuration

The JSON files map experiment names to settings. The runner adds the `CODEX_`
prefix. Settings are loaded once when the bot starts. `AUTO=0` disables the
switch between multiplayer and duel policy, enabling the original parameter
experiments to be rerun. `AUTO=1` recalculates the policy from the surviving
opponents each turn, including when a multiplayer game becomes a duel.

Core settings are `HOLD` (combat strength-to-production threshold), `INT_MAX`
(interior strength percentile), `DEGRADE` (routing distance penalty),
`PROD_POWER` (production exponent in target value), `WALLS` (preserve neutral
walls), and `STRATEGIC` (the additional combat-waiting rule). `DUEL_HOLD` and
`DUEL_INT_MAX` configure the automatic duel policy. That policy releases walls,
disables extra strategic waiting, and uses the original production exponent and
routing penalty. The original local danger avoidance and overflow checks remain
active in both modes.

The JSON experiments preserve unsuccessful alternatives as well as successful
ones. A separate one-turn combat search was tested and rejected; its code is not
part of the released bot. Read `RESULTS.md` for measured results and limitations.
