"""Integration checks for the reward, simulator, protocol, and resumable evolution.

Run: python3 -m unittest discover -s evolution -p 'test_*.py' -v
"""
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from train import EVALUATOR, HERE, ROOT, PARAMS, evaluate, fresh, reproduce, write_policy

sys.path.insert(0, str(ROOT))
from halite import GameState, generate_map

BOT = ROOT / "starter-kit/rust/target/release/evo_bot"


def flatten(grid):
    return [v for row in grid for v in row]


def encode(frame):
    owners = frame["owners"]
    runs = []
    i = 0
    while i < len(owners):
        end = i + 1
        while end < len(owners) and owners[end] == owners[i]:
            end += 1
        runs.extend((end - i, owners[i]))
        i = end
    return " ".join(map(str, runs + frame["strengths"]))


class EvolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(["cargo", "build", "--release", "--manifest-path", str(HERE / "Cargo.toml")], check=True, capture_output=True)
        subprocess.run(["cargo", "build", "--release", "--manifest-path", str(ROOT / "starter-kit/rust/Cargo.toml"), "--bin", "evo_bot"], check=True, capture_output=True)

    def test_mutation_is_reproducible_preserves_elites_and_parents(self):
        rng = random.Random(7)
        parents = [fresh(rng, str(i)) for i in range(16)]
        before = copy.deepcopy(parents)
        a = reproduce(random.Random(9), parents, 1, 32)
        b = reproduce(random.Random(9), parents, 1, 32)
        self.assertEqual(a, b)
        self.assertEqual(parents, before)
        self.assertEqual(a[:2], parents[:2])
        self.assertTrue(any(p["weights"] != parents[0]["weights"] for p in a[2:]))
        self.assertTrue(all(len(p["weights"]) == PARAMS for p in a))

    def test_simulator_and_protocol_are_frame_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            # An evolved policy exercises expansion/combat; a random one adds varied moves.
            a = ROOT / "starter-kit/rust/weights/evo_bot.evo"
            b = directory / "random.evo"
            write_policy(b, fresh(random.Random(81), "test")["weights"])
            for seed, size, swap in [(14, 12, 0), (555, 20, 1)]:
                with self.subTest(seed=seed, swap=swap):
                    path = directory / "replay.json"
                    result = evaluate([(a, b, seed, size, swap)], directory, 1, path)[0]
                    replay = json.loads(path.read_text())
                    w, h, prod, strength, starts = generate_map(size, size, 2, random.Random(seed))
                    game = GameState(w, h, prod, strength, starts)
                    self.assertEqual(replay["production"], flatten(prod))
                    deaths = [None] * 3
                    for t, frame in enumerate(replay["frames"]):
                        self.assertEqual(frame["owners"], flatten(game.owner))
                        self.assertEqual(frame["strengths"], flatten(game.strength))
                        stats = game.stats()
                        for p in (1, 2):
                            if deaths[p] is None and stats[0][p] == 0:
                                deaths[p] = t
                        if t + 1 == len(replay["frames"]):
                            break
                        moves = {p: {(i % w, i // w): d for i, d in enumerate(frame["moves"])
                                     if frame["owners"][i] == p} for p in (1, 2)}
                        game.step(moves)
                    self.assertEqual(result["turns"], len(replay["frames"]) - 1)
                    self.assertTrue(sum(d is None for d in deaths[1:]) <= 1 or result["turns"] == int(10 * (w*h)**.5))
                    me = replay["candidate_player"]
                    key = lambda p: (deaths[p] is None, deaths[p] or 0, stats[0][p], stats[1][p])
                    expected = 1 if key(me) > key(3-me) else 0 if key(me) < key(3-me) else .5
                    self.assertEqual(expected, result["score"])
                    # Feed every recorded observation through the actual standalone bot.
                    for player, weights in [(me, a), (3-me, b)]:
                        frames = replay["frames"][:-1]
                        wire = f"{player}\n{w} {h}\n" + " ".join(map(str, replay["production"])) + "\n"
                        wire += encode(frames[0]) + "\n" + "\n".join(map(encode, frames)) + "\n"
                        response = subprocess.run([str(BOT)], input=wire, text=True, capture_output=True, check=True,
                                                  env={**os.environ, "EVO_WEIGHTS": str(weights)}, timeout=20)
                        lines = response.stdout.splitlines()
                        self.assertEqual(lines[0], "EvoBot")
                        self.assertEqual(len(lines), len(frames) + 1)
                        for line, frame in zip(lines[1:], frames):
                            tokens = list(map(int, line.split()))
                            moves = {(tokens[i], tokens[i+1]): tokens[i+2] for i in range(0, len(tokens), 3)}
                            for i, o in enumerate(frame["owners"]):
                                if o == player:
                                    self.assertEqual(moves.get((i % w, i // w), 0), frame["moves"][i])

    def test_exact_ties_and_parallel_order(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            zero = directory / "zero.evo"
            write_policy(zero, [0] * PARAMS)
            jobs = [(zero, zero, seed, 8, swap) for seed in (1, 2, 3) for swap in (0, 1)]
            a = evaluate(jobs, directory, 1)
            b = evaluate(jobs, directory, 3)
            self.assertEqual(a, b)
            self.assertTrue(all(r["score"] == .5 for r in a))

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "full", Path(temp) / "resumed"
            base = [sys.executable, str(HERE / "train.py"), "--population", "4", "--opponents", "2", "--maps", "1", "--sizes", "8", "--workers", "2", "--seed", "123"]
            for path, count, resume in [(a, 3, False), (b, 1, False), (b, 3, True)]:
                subprocess.run(base + ["--run", str(path), "--generations", str(count)] + (["--resume"] if resume else []),
                               check=True, capture_output=True, text=True)
            sa, sb = [json.loads((p / "state.json").read_text()) for p in (a, b)]
            self.assertEqual(sa["population"], sb["population"])
            self.assertEqual(sa["rng"], sb["rng"])
            self.assertEqual(sa["games"], sb["games"])
            for ga, gb in zip(sa["history"], sb["history"]):
                ga.pop("seconds"); gb.pop("seconds")
                self.assertEqual(ga, gb)
            self.assertEqual((a / "champion.evo").read_bytes(), (b / "champion.evo").read_bytes())


if __name__ == "__main__":
    unittest.main()
