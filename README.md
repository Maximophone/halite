# Halite I clone

A local re-creation of **Halite**, the first season (2016 to 2017) of the AI programming
competition run by Two Sigma and Cornell Tech. Write a bot, pit it against others, and
watch the replay.

Only Python 3 is needed (no packages to install).

## Quick start

```bash
# Your bot vs the sample bot, random map size
python3 halite.py "python3 starter-kit/python/MyBot.py" "python3 starter-kit/python/ImprovedBot.py"

# Fixed size and seed, up to 6 players
python3 halite.py -d "30 30" -s 42 "python3 starter-kit/python/MyBot.py" "python3 starter-kit/python/RandomBot.py"
```

Each game writes to `replays/`:

- a `.html` file: double-click it to watch the game (the replay is embedded)
- a `.hlt` file: the raw replay (JSON), which you can drop onto `viewer.html`

Whatever your bot writes to stderr is saved in `logs/bot-N.log`. Never print to stdout:
that is the channel to the game.

To generate a sample game to look at: `python3 halite.py -d "40 40" -s 2016 -o examples "python3 starter-kit/python/ImprovedBot.py" "python3 starter-kit/python/ImprovedBot.py" "python3 starter-kit/python/RandomBot.py" "python3 starter-kit/python/MyBot.py"`

## Files

| File | What it is |
|---|---|
| `halite.py` | Game environment: map generator, rules, bot protocol, replays |
| `viewer.html` | Replay viewer (play, scrub, pan the wrap-around map, stats charts) |
| `tournament.py` | Runs many games and reports wins and average rank |
| `arena.py`, `arena.html` | Endless tournament server with a leaderboard and a browsable list of every game |
| `test_rules.py` | Checks for the combat, movement and production rules |
| `starter-kit/python/` | `hlt.py` and `networking.py` (original API), `MyBot.py`, `RandomBot.py`, `ImprovedBot.py`, `PredatorBot.py` |

## Options

```
python3 halite.py [-d "W H"] [-s SEED] [-t] [-q] [-o DIR] [--no-replay] [--no-html] [--turns N] BOT [BOT ...]
  -d   map size (default: random square from 20 to 50)
  -s   map seed, to replay the same map
  -t   no timeouts (lets you debug a slow bot)
  -q   print only a JSON summary
```

```bash
python3 tournament.py -n 20 "python3 starter-kit/python/MyBot.py" "python3 starter-kit/python/ImprovedBot.py"
python3 tournament.py -n 20 --players 4 "python3 a.py" "python3 b.py" "python3 c.py" "python3 d.py" "python3 e.py"
```

## The arena

```bash
python3 arena.py
```

Then open http://localhost:8000. The arena plays games forever, several at once (half your CPU
cores by default, `--workers N` to change), between every `starter-kit/python/*Bot.py`. A new bot
file dropped in that folder joins on the next game. The page shows an Elo-style leaderboard, the
games in progress and every finished game; click a bot to filter, click Watch to open a replay.
Games mix 2, 3, 4 and 6 players on random maps.

Bots in other languages: anything that speaks the protocol works.

```bash
python3 arena.py --bot "RustBot=./bots/rust/target/release/rustbot" --bot "CBot=./bots/c/cbot"
```

Everything is stored in `arena/` (`games.jsonl` plus replays). Stop with Ctrl-C and restart any
time: ratings are rebuilt from `games.jsonl`. Only the latest 1000 replays are kept (`--keep N`).
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

The Python starter kit keeps the original API:

```python
from hlt import *
from networking import *

myID, gameMap = getInit()
sendInit("MyPythonBot")

while True:
    gameMap = getFrame()
    moves = []
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            loc = Location(x, y)
            site = gameMap.getSite(loc)        # .owner .strength .production
            if site.owner == myID:
                moves.append(Move(loc, STILL))
    sendFrame(moves)
```

Helpers on `gameMap`: `getSite(loc, direction)`, `getLocation(loc, direction)`,
`getDistance(a, b)` (wrap-aware), `getAngle(a, b)`.

`ImprovedBot.py` is a good first target: it captures the best-value neighbouring site,
waits until it has 5x its production in strength, and sends interior pieces to the border.
Classic next steps from the original competition: route interior strength toward the
weakest point of the border, avoid merging past 255, and use overkill when fighting
(move into the empty gap between you and an enemy so your piece hits all their pieces at once).

## How close is it to the original?

The rules follow the description in the creators' paper (Spector and Truell,
*The Design and Implementation of Modern Online Programming Competitions*, 2017) and the
original starter kit protocol. Two parts are re-implementations rather than copies:

- **Map generator:** smooth noise mirrored into one tile per player, with each start at its tile centre, so starts are fair and evenly spaced.
  Maps look and play similarly but are not byte-identical to the original generator.
  The map may shrink slightly so the tiles divide it evenly (e.g. 40x40 with 6 players is 39x40).
- **Edge cases** such as several zero-strength pieces of different players on one site
  (nobody keeps it) and ejected bots (their sites turn neutral).
