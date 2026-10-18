#!/usr/bin/env python3
"""Plain-assert checks of the arena's skill ratings. Run: python3 test_ratings.py"""

import json
import os
import shutil
import tempfile

import arena
from arena import MU, SIGMA, apply_game, conservative, rate


def game(gid, *bots):
    """A finished game where `bots` finish in the order given."""
    return {"id": gid, "time": 0, "duration": 0, "seed": gid, "width": 20, "height": 20,
            "turns": 100, "replay": None,
            "players": [{"bot": b, "name": b, "rank": r + 1, "territory": 0, "strength": 0,
                         "status": "alive"} for r, b in enumerate(bots)]}


def test_winner_rises():
    skills, stats = {}, {}
    for k in range(50):
        apply_game(game(k, "Strong", "Weak"), skills, stats)
    assert skills["Strong"][0] > MU > skills["Weak"][0]
    assert conservative(skills["Strong"]) > conservative(skills["Weak"])
    # Still rising after many wins, but by less each time.
    before = skills["Strong"][0]
    apply_game(game(50, "Strong", "Weak"), skills, stats)
    assert skills["Strong"][0] > before
    assert stats["Strong"]["wins"] == 51 and stats["Weak"]["wins"] == 0


def test_uncertainty_shrinks():
    skills, stats = {}, {}
    sigmas = []
    for k in range(100):
        bots = ["New", "A", "B"] if k % 2 else ["A", "New", "B"]
        apply_game(game(k, *bots), skills, stats)
        sigmas.append(skills["New"][1])
    assert sigmas[0] < SIGMA
    assert all(b < a for a, b in zip(sigmas, sigmas[1:]))
    assert sigmas[-1] < SIGMA / 2


def test_four_players_all_update():
    skills, stats = {}, {}
    apply_game(game(1, "P1", "P2", "P3", "P4"), skills, stats)
    mus = [skills[b][0] for b in ("P1", "P2", "P3", "P4")]
    assert all(m != MU for m in mus)
    assert mus[0] > mus[1] > mus[2] > mus[3]
    assert all(skills[b][1] < SIGMA for b in skills)
    # Equal priors: the mean changes add up to zero.
    assert abs(sum(mus) - 4 * MU) < 1e-9


def test_ties_and_six_players():
    new = rate([(MU, SIGMA)] * 2, [1, 1])
    assert abs(new[0][0] - MU) < 1e-9 and abs(new[1][0] - MU) < 1e-9
    assert new[0][1] < SIGMA
    new = rate([(MU, SIGMA)] * 6, [1, 2, 3, 4, 5, 6])
    assert all(a[0] > b[0] for a, b in zip(new, new[1:]))


def test_duplicate_seats():
    # Fewer bots than seats: a bot gets the sum of its seats' changes, and
    # two seats tell more than one.
    skills, stats = {}, {}
    apply_game(game(1, "X", "Y", "X"), skills, stats)
    assert abs(skills["X"][0] + skills["Y"][0] - 2 * MU) < 1e-9
    assert skills["X"][1] < skills["Y"][1] < SIGMA
    assert stats["X"]["games"] == 2


def test_snapshot_rebuild():
    """Compaction folds games into state.json and a restart gives the same skills."""
    tmp = tempfile.mkdtemp()
    try:
        old = arena.COMPACT_SLACK
        arena.COMPACT_SLACK = 5
        a = arena.Arena(tmp, {}, False, keep=10, history=10)
        order = ["A", "B", "C", "D"]
        for k in range(1, 40):
            g = game(k, *order[k % 4:] + order[:k % 4])
            a.finish(g)
        with open(os.path.join(tmp, "state.json")) as f:
            base = json.load(f)
        assert base["games"] > 0 and set(base["skill"]) == set(order)
        b = arena.Arena(tmp, {}, False, keep=10, history=10)
        for bot in order:
            assert all(abs(x - y) < 1e-9 for x, y in zip(a.skills[bot], b.skills[bot]))
        # Old Elo snapshot: skills restart from the listed games, totals kept.
        with open(os.path.join(tmp, "state.json")) as f:
            base = json.load(f)  # b compacted again on load
        base.pop("skill")
        base["ratings"] = {bot: 1500.0 for bot in order}
        with open(os.path.join(tmp, "state.json"), "w") as f:
            json.dump(base, f)
        c = arena.Arena(tmp, {}, False, keep=10, history=10)
        assert "ratings" not in c.base and set(c.skills) == set(order)
        assert c.stats["A"]["games"] == a.stats["A"]["games"]
        board = c.state(10, "")["leaderboard"]
        assert [r["rating"] for r in board] == sorted((r["rating"] for r in board), reverse=True)
    finally:
        arena.COMPACT_SLACK = old
        shutil.rmtree(tmp)


def test_halite_matchmaking():
    """Near-equals meet more than distant bots, uncertain and idle bots get seats."""
    import random
    rng = random.Random(1)
    names = ["B%02d" % i for i in range(15)]
    skills = {b: (10.0 + 2 * i, 0.7) for i, b in enumerate(names)}
    skills["B07"] = (MU, SIGMA)          # a new bot
    meet = {}
    seats = dict.fromkeys(names, 0)
    last = {b: 0 for b in names}
    for gid in range(1, 20001):
        lineup = arena.halite_lineup(rng, names, 2, skills, last)
        assert len(lineup) == 2 and lineup[0] != lineup[1]
        for b in lineup:
            seats[b] += 1
            last[b] = gid
        meet[frozenset(lineup)] = meet.get(frozenset(lineup), 0) + 1
    near = meet.get(frozenset(["B13", "B14"]), 0)
    far = meet.get(frozenset(["B00", "B14"]), 0)
    assert near > 5 * far > 0, (near, far)
    assert seats["B07"] == max(seats.values())
    assert min(seats.values()) > 20000 * 2 / 15 / 3
    # More seats than bots: everyone plays, some twice.
    lineup = arena.halite_lineup(rng, names[:3], 6, skills, last)
    assert len(lineup) == 6 and set(lineup) == set(names[:3])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print("%d rating checks passed" % len(tests))
