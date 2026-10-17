#!/usr/bin/env python3
"""Rule checks for the engine. Run: python3 test_rules.py"""

from halite import GameState, STILL, NORTH, EAST, SOUTH, WEST


def make(w, h, cells, prod=0):
    """cells: {(x, y): (owner, strength)}; everything else neutral strength 0."""
    production = [[prod] * w for _ in range(h)]
    strength = [[0] * w for _ in range(h)]
    g = GameState(w, h, production, strength, [])
    g.num_players = max([o for o, _ in cells.values()] + [1])
    for (x, y), (o, s) in cells.items():
        g.owner[y][x] = o
        g.strength[y][x] = s
    return g


def at(g, x, y):
    return g.owner[y][x], g.strength[y][x]


def test_still_gains_production_and_caps():
    g = make(5, 5, {(1, 1): (1, 250)}, prod=10)
    g.step({})
    assert at(g, 1, 1) == (1, 255), at(g, 1, 1)


def test_move_leaves_zero_piece_and_captures():
    g = make(5, 5, {(1, 1): (1, 50), (2, 1): (0, 20)})
    g.step({1: {(1, 1): EAST}})
    assert at(g, 1, 1) == (1, 0)
    assert at(g, 2, 1) == (1, 30)


def test_failed_capture_weakens_neutral():
    g = make(5, 5, {(1, 1): (1, 10), (2, 1): (0, 30)})
    g.step({1: {(1, 1): EAST}})
    assert at(g, 2, 1) == (0, 20)
    assert at(g, 1, 1) == (1, 0)


def test_equal_strength_neutral_both_die():
    g = make(5, 5, {(1, 1): (1, 30), (2, 1): (0, 30)})
    g.step({1: {(1, 1): EAST}})
    assert at(g, 2, 1) == (0, 0)


def test_merge_caps_at_255():
    g = make(5, 5, {(0, 1): (1, 200), (2, 1): (1, 200), (1, 1): (1, 0)})
    g.step({1: {(0, 1): EAST, (2, 1): WEST}})
    assert at(g, 1, 1) == (1, 255)


def test_overkill_damages_all_adjacent_enemies():
    # Player 1 piece of 100 at center, three enemy pieces of 40 around it.
    cells = {(2, 2): (1, 100), (2, 1): (2, 40), (3, 2): (2, 40), (2, 3): (2, 40)}
    g = make(5, 5, cells)
    g.step({})
    assert at(g, 2, 2) == (0, 0)  # took 3 x 40 = 120 damage: dies
    for loc in [(2, 1), (3, 2), (2, 3)]:
        assert at(g, *loc)[0] == 0  # each took 100 damage


def test_neutral_does_not_fight_adjacent():
    g = make(5, 5, {(1, 1): (1, 5), (2, 1): (0, 200)})
    g.step({})
    assert at(g, 1, 1) == (1, 5)
    assert at(g, 2, 1) == (0, 200)


def test_wraparound():
    g = make(4, 4, {(0, 0): (1, 50)})
    g.step({1: {(0, 0): WEST}})
    assert at(g, 3, 0) == (1, 50)
    g.step({1: {(3, 0): NORTH}})
    assert at(g, 3, 3) == (1, 50)


def test_serialize_roundtrip_shape():
    g = make(3, 2, {(0, 0): (1, 9), (1, 0): (1, 8)})
    assert g.serialize().split() == ["2", "1", "4", "0", "9", "8", "0", "0", "0", "0"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print("%d rule checks passed" % len(tests))
