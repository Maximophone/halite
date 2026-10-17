# Building strong Halite bots: notes for the next AI

You are probably here to make a bot stronger than ApexAlpha. I built ApexAlpha, and PredatorBot
before it, over one long session. This is what I wish someone had told me on the first day. It is
less about Halite tricks (those are at the end) and more about how to work: where the time went,
where my judgement was wrong, and which habits turned hours of guessing into minutes of fixing.

Read CLAUDE.md first for the rules, the protocol and the project layout. The tools named below
are in `tools/`; each one explains itself with `-h`:

- `tools/eval_bot.py`: a candidate against a fixed opponent pool, many games in parallel, fixed
  seeds, win rate, average rank and errors per format.
- `tools/solo_bench.py`: expansion speed with no opponent (production at turns 60, 100, 150).
- `tools/trace_game.py`: one game, each player's territory, production and strength over time,
  plus the replay to watch.

## The short version

1. Build the measuring instrument before the bot. Most of my progress came after I could measure
   something precisely, not after I had a clever idea.
2. Split "win rate" into parts you can measure separately. Win rate is noisy and mixes
   everything; expansion speed alone is deterministic and told me far more.
3. When a parameter sweep comes back flat, your theory of the problem is wrong. Stop sweeping
   and instrument the bot.
4. Look at the actual numbers inside the bot (the values it computes, where its strength sits)
   before reasoning about strategy. Both of my worst bugs were visible in one debug line.
5. Keep every change honest with ablations, fresh seeds and enough games.

## 1. Measurement comes first

My first PredatorBot lost 7 of 10 games, and I spent a while reading code and guessing. What
actually moved things was tooling, in this order of value:

**A parallel evaluation harness with fixed seeds.** Run many games at once on the Rust engine
(a 50x50 four-player game takes about 0.4 s), with the same seeds and seat shuffles for every
candidate, so two versions are compared on identical maps. Report win rate and average rank
per format (1v1, 4-player, 6-player) and, always, the number of games where your bot crashed or
timed out. A crashing bot looks exactly like a weak bot. My first "improvement" to PredatorBot
went from 15/20 to 1/20 wins because of a Python dict modified during iteration; the error
column would have told me instantly.

**Know your noise.** At 60 games, win rates move by about plus or minus 6 points from luck
alone. I made decisions on 3-point differences early on and some of them were noise. Use 100+
games for anything you intend to keep, and re-check the final candidate on seeds you never
tuned on (I used `--seed 2` for that: it confirmed 89% in 1v1).

**A deterministic proxy metric.** The single best tool I built was a solo benchmark: the bot
plays alone on 40 fixed maps and I record the production it owns at turns 60, 100 and 150. No
opponent, no randomness, so a 2% change is real. Once I had it, I found in an hour what weeks of
win-rate tuning would not have found: a gathering planner (+10%) and a ranking exponent (+7% more,
then +20% over PredatorBot in total). In FFA, expansion speed decides who eats whom, so this
proxy translated directly into wins (6-player wins went from 30% to 48%).

**Tunable constants without rebuilding.** Every constant in ApexAlpha reads an `APEX_*`
environment variable. A sweep is then a shell loop over commands like
`APEX_DANGER=7 ./apex_bot`, and the engine passes the variable through because bots run through
a shell. This made sweeps cheap enough that I actually ran them.

**Name and freeze versions.** I copied each build to `bots/apex_vN` before changing anything,
so every comparison was against a fixed binary. The arena does the same at a larger scale:
ApexAlpha is frozen, and the next version should be a new file with a new name so both keep
their own rating.

## 2. Decompose the objective

"Win more games" was too blunt to steer by. What worked was asking which part of the game a loss
came from, then measuring that part on its own:

- **Economy:** production owned over time (solo benchmark, or per-player curves from a replay).
- **Where the strength sits:** border versus interior strength, pieces that want to move but
  cannot, pieces at the 255 cap.
- **Combat:** territory lost after first contact while total strength is similar.

A small trace script that prints each player's territory, production and strength every 20
turns of one game answered "economy or combat?" in seconds. For example, it showed ApexBot
holding three times PredatorBot's territory with the same production: it was expanding into poor
land, and then losing it to concentrated attacks. That one table redirected the whole effort.

## 3. Instrument; do not theorise

Twice I had a confident explanation that was wrong:

- I thought the fight's pull through the value field was too strong, so I added a separate decay
  for it and swept it. The results were identical to the last decimal. That should have been the
  alarm: a parameter that changes nothing is not touching the problem.
- I thought combat was too reckless and swept a caution weight. Flat again.

Then I added one debug line (APEX_DEBUG, written to the bot's stderr log) and the answer was
right there: interior pieces held 18,000 strength, the border held 2,000, and the maximum value
in the routing field was 18 while a good target is worth about 0.5. The culprit was empty neutral
sites: the formula gave a strength-0 site its full production as value (`prod / 0` became
`prod + 1`). A free site needs no strength sent to it, yet it pulled every interior piece within
about 20 steps toward it. Ten minutes from that debug line to the fix.

Rules I now follow:

- When a sweep is flat, stop and log the internals: counts of pieces by role, their strength,
  the field's min and max, how many pieces move toward which kind of target.
- Check that every value feeding a max-propagated field lives on the same scale. Print the max.
  Anything with a division needs a floor (ApexAlpha uses `prod / max(strength, 15)`).
- Watch the replay, but confirm who is who first. I once analysed the wrong colour for a whole
  paragraph: player 1 is yellow, player 2 is blue, and the seat order depends on the command
  order.

## 4. Taste: what made changes stick

- **Ablate.** When a feature helps, switch off each part of it once. The defence had two parts
  (keep nearby pieces out of expansion, and pull strength to the front); turning each off showed
  both mattered in 1v1 (85% with both, 78% and 71% without one or the other).
- **Write down what did nothing.** CLAUDE.md lists the dead ends (fight caution, fight pull,
  hunting weak players, most expansion constants once gathering existed). Negative results stop
  the next session from re-running the same sweeps.
- **Prefer structure over constants.** Before gathering, no expansion constant moved the solo
  benchmark by more than 2%. The limit was the algorithm, not its tuning. If ten knobs are flat,
  the next gain is a new mechanism.
- **Respect determinism.** Ties broken by index, stable sorts, no randomness. It made Python to
  Rust ports provably identical (40 of 40 replays equal) and made the solo benchmark exact.
- **Do not trust Elo between near-identical bots.** The Python and Rust PredatorBots drifted 200
  points apart in the arena while finishing ahead of each other exactly 52% of the time. Use
  head-to-head counts for close comparisons.

## 5. What I learnt about this game

Mechanics that drive strategy:

- **Overkill favours concentration.** A piece deals its full strength to every enemy next to it.
  One 255 piece walking into a field of 20-strength pieces destroys them all. Large, territory
  thin with tiny pieces is a liability, not an asset. Raising the minimum strength before a piece
  moves from 8 to 25 took 1v1 from 57% to 70%.
- **Expansion compounds.** Early production is worth the most. Plan captures: for each frontier
  target, how many turns until enough strength can arrive from pieces up to 8 steps away (a piece
  d steps away arrives in d turns; closer pieces grow while they wait). Rank targets by value
  divided by turns to the power 0.25. Exponent 1 (pure speed) was clearly worse: patience for
  good land beats grabbing whatever is quickest. Re-plan every turn.
- **Walls are free defence.** A neutral site touching an enemy costs whoever breaks it. Leaving
  them standing in FFA lets the neighbours bleed on each other.
- **Defend before the breach.** Keep pieces within about 5 steps of an enemy out of expansion and
  let the front pull interior strength. This was the largest single combat gain (73% to 88% 1v1).
- **FFA is about not being eaten.** Ranking is by survival first. Whoever eats the weakest player
  early snowballs. ImprovedBot ranks last in 4-player games behind random bots, because it grows
  fast and becomes the first meal.

What ApexAlpha still does badly (start here for ApexBeta):

- Combat micro ignores what the enemy will do this turn. It scores moves as if enemies stay
  still. One-ply prediction of enemy moves at the front is the obvious next step.
- 4-player results (53% on fresh seeds) lag 1v1 and 6-player. Look at who it fights and when.
- It has never been tuned against itself. All tuning was against PredatorBot, so some choices may
  be specific to that opponent. Add ApexAlpha to the evaluation pool before claiming progress.
- The opening (first 30 turns) is handled by the general planner. A searched opening could win
  a few percent of production early, which compounds.

## 6. A workflow that would have saved me half the time

1. Use (or extend) `tools/eval_bot.py` and `tools/solo_bench.py` before writing bot logic.
2. Get a baseline for every existing bot on both.
3. Change one thing; build to a new versioned binary; run solo (for economy changes) or the
   harness (for combat changes) at 100 games.
4. If the result is flat, instrument before trying a second variant.
5. Before freezing a version, ablate its parts and confirm on fresh seeds.
6. Write the result, including what failed, into CLAUDE.md.

Good luck. Make ApexBeta beat ApexAlpha, not just PredatorBot.
