# Evolution lab

`EvoBot` learns exclusively through **mutation, selection, and self-play**. It starts
with independently random weights. There is no imitation, gradient descent,
scripted action selection, or reward for a hand-picked behavior.

Open [the saved experiment](report.html) directly in a browser for training curves,
population fitness and ancestry, mutation/diversity plots, policy weights, and a
playable final-generation replay. The live dashboard also loads a replay for every
generation. All data is from actual matches.

## Run it

From the repository root (Python 3 and Rust; **no third-party dependencies**):

```bash
# The bundled champion works immediately after building.
cargo build --release --manifest-path starter-kit/rust/Cargo.toml --bin evo_bot
python3 halite.py starter-kit/rust/target/release/evo_bot starter-kit/rust/target/release/improved_bot

# Start a new population from random initialization. Builds the evaluator automatically.
python3 evolution/train.py --run evolution/runs/my-run --generations 240

# Live dashboard, bound to localhost only.
python3 evolution/dashboard.py --run evolution/runs/my-run --port 8092

# Continue the same population to 500 total generations, with a portable report.
python3 evolution/train.py --run evolution/runs/my-run --resume --generations 500 --export evolution/my-report.html

# Play a checkpoint without rebuilding.
python3 halite.py "EVO_WEIGHTS=evolution/runs/my-run/champion.evo starter-kit/rust/target/release/evo_bot" starter-kit/rust/target/release/improved_bot
```

For a larger selection sample:

```bash
python3 evolution/train.py --run evolution/runs/larger --population 64 --opponents 5 --maps 6 --sizes 16 20 24 32 --generations 500 --workers 6
```

The bundled experiment is `--seed 42 --population 32 --opponents 3 --maps 2
--sizes 12 16 20 24 --generations 240`. Runs are ignored by git; the frozen bot
checkpoint, experiment provenance, and saved report are included. The bot is
automatically discovered by the existing arena after it is built.

### Bundled experiment results

The 240-generation run played 92,160 training matches, 4,320 fixed-panel evaluation
matches, and 240 replay matches. The final policy scored 18/18 against the founding
panel. Post-training checks against established bots put that number in context:

| Evaluation | Wins | Games | Crashes/timeouts |
|---|---:|---:|---:|
| 1v1 vs ImprovedBot | 14 (23.3%) | 60 | 0 |
| 1v1 vs PredatorBot | 7 (11.7%) | 60 | 0 |
| 4-player vs three ImprovedBots | 0 | 12 | 0 |

These use `tools/eval_bot.py --seed 867`, default random map sizes, and randomized
seats. They were run after freezing the final checkpoint and did not influence
training. This is a working proof of learning from scratch, not a top arena bot.
See `experiment.json` for configuration and the checkpoint hash.

## What evolves

The architecture is deliberately small: **24 features → 8 softsign hidden neurons
→ one action score**, plus a learned linear shortcut. There are **232 real-valued
weights**. One network scores all five actions at every owned cell; the highest
score wins. Exact score ties prefer the first action in stay/north/east/south/west
order. Each cell acts on the same pre-move map, matching simultaneous game rules.

Inputs include source and target strength/production; relative ownership; adjacent
friendly/enemy strength and counts; wraparound distance to non-owned territory and
enemies; changes in those distances; elapsed time; territory and production balance;
potential merge overflow; and two-step strength/production. They encode observable
geometry and game quantities. They do **not** prescribe when to attack, wait, merge,
or expand. No move is filtered or overridden by a heuristic. The shared action scorer
works on arbitrary map sizes and player IDs. An engineered feature representation
is the main inductive bias; this is not raw-pixel learning.

This compact policy makes black-box evolution inexpensive. The fixed architecture
does not evolve topology. Mutation alone avoids needing to align interchangeable
hidden neurons for crossover.

## Selection and self-play

1. Evaluate every candidate against the **same frozen opponent panel**, map seeds,
   map sizes, and both starting seats. Defaults: 3 opponents × 2 maps × 2 seats =
   12 matches per candidate, 384 training matches per generation.
2. Fitness is the mean outcome: win = 1, loss = 0, exact tie = 0.5. Outcomes follow
   survival, elimination time, territory, and strength, as in the engine, except
   exact ties are shared rather than broken by player ID. Territory and other
   telemetry never contribute additional fitness or break selection ties.
3. Keep two elites unchanged. Choose parents uniformly from the top quarter
   (at least two). Break equal fitness randomly using the saved seeded RNG.
4. Mutate the parent's mutation scale multiplicatively, clamped to 0.015–0.6.
   Add Gaussian noise to 30% of weights, or to all weights for 20% of children.
   Clamp weights to ±8. Six percent of new slots are independent random immigrants.
5. Archive the generation champion. The next panel contains the latest champion,
   a randomly sampled historical champion, and random members of the previous
   population. With custom opponent counts, the panel is truncated/extended in
   that order. This provides memory and varied contemporaries.

Games use `engine/src/game.rs` and `engine/src/pyrandom.rs` directly, in process.
There is no approximate training simulator. Games run in parallel but are returned
in input order; changing worker count does not change selection.

## Reading the results

- **Training scores are relative to a changing panel.** They are not Elo and should
  not be interpreted as an absolute skill curve. Twelve games is a noisy fitness
  estimate; increase `--maps` and `--opponents` for more reliable selection.
- **Fixed-panel evaluation** freezes the top three generation-zero policies. Each
  champion plays 18 games on fixed unseen seeds at sizes 16, 24, 32 and both seats.
  These results are logged only; they never influence reproduction, mutation,
  opponent sampling, or checkpoint selection. This is a weak founding baseline,
  not evidence of mastery or a statistically independent generalization estimate.
- `champion.evo` is the **last generation's training champion**, not a checkpoint
  cherry-picked on evaluation scores. Non-monotonic skill and cycling are possible.
- The replay uses a separate fixed seed against the preceding champion. Its actions,
  ownership, strengths, and economies are recorded from the actual engine.
- Training is two-player. The executable accepts up to six players, but it has not
  learned multiplayer-specific strategies. External benchmark results, when
  recorded in `experiment.json`, are evaluation only and never training opponents.
- Weight bars visualize only the linear shortcut. Hidden-layer interactions can
  change or reverse their apparent behavioral interpretation.

## Checkpoints and recovery

Each run stores an atomic `state.json`: population weights, individual parents and
mutation scales, opponent panel, champion archive, histories, configuration, and
the full Python random-generator state. Checkpoints and replays live in numbered
generation directories. `--resume` uses the saved configuration, including the
original seed; only worker count, total generations, and report destination can
change. Interrupted work since the last completed generation is deterministically
replayed. A process lock prevents two trainers writing the same run.

`HALITE_EVO_V1` checkpoints contain 232 decimal weights. Both evaluator and bot
load the exact same format and compile the same `src/policy.rs`. The bundled bot
embeds `starter-kit/rust/weights/evo_bot.evo`; `EVO_WEIGHTS` overrides it at runtime.
Invalid/truncated/nonfinite files fail visibly, with no random fallback.

Run manifests currently store absolute checkpoint paths, so keep a resumable run
at its original location. The exported report and individual `.evo` policies are
portable. Run storage grows with generation count (including one full replay per
generation); it is not automatically pruned.

## Verification

```bash
cargo test --manifest-path evolution/Cargo.toml
python3 -m unittest discover -s evolution -p 'test_*.py' -v
```

Tests cover malformed weights, wraparound and player-ID symmetry, deterministic
mutation and untouched elites, exact ties, worker ordering, resume equivalence,
full replay parity with the Python reference engine, and every standalone bot move
against the in-process policy for recorded games. Dashboard controls are also
checked interactively: historical generations, replay scrubbing, and map layers.

Files: `train.py` handles evolution and recovery; `src/policy.rs` is inference;
`src/main.rs` batches games; `src/sim.rs` handles termination; `dashboard.py` and
`dashboard.html` show the run. The entire experiment runs on CPU.
