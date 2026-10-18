"""A release reset survives restarts and compaction without rerating opponents."""
import copy
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import arena
from test_ratings import game


def test_reset():
    with tempfile.TemporaryDirectory() as directory:
        a = arena.Arena(directory, {}, False, keep=10, history=5)
        for i in range(1, 16):
            a.finish(game(i, "CodexBot", "A", "B") if i % 2 else game(i, "B", "A"))
        a._compact(force=True)
        before = copy.deepcopy((a.skills, a.stats))
        total = a.state(100, "")["total_games"]
        for paused, running in [(False, {}), (True, {16: {}})]:
            a.paused, a.running = paused, running
            try:
                a.reset_bot("CodexBot")
                raise AssertionError("Unsafe reset accepted")
            except ValueError:
                pass
        a.paused, a.running, a.next_id = True, {}, 16
        a.analytics()  # Populate cache before the reset.
        reset = a.reset_bot("CodexBot")
        assert reset["id"] == 16
        for bot in ("A", "B"):
            assert a.skills[bot] == before[0][bot]
            assert a.stats[bot] == before[1][bot]
        assert a.skills["CodexBot"] == [arena.MU, arena.SIGMA]
        assert a.stats["CodexBot"]["games"] == 0
        assert a.state(100, "")["total_games"] == total
        assert a.state(100, "CodexBot")["matching_games"] == 0
        chart = a.analytics()
        assert "CodexBot" not in chart["bots"]
        assert "CodexBot" not in chart["h2h"]
        assert "CodexBot" not in chart["ratings"]
        for g in a.state(100, "")["games"]:
            assert not any(p["bot"] == "CodexBot" for p in g["players"])

        b = arena.Arena(directory, {}, False, keep=10, history=5)
        assert b.skills == a.skills and b.stats == a.stats
        assert b.resets == a.resets and b.next_id == 17
        b.finish(game(17, "CodexBot", "A", "B"))
        assert b.stats["CodexBot"]["games"] == 1
        assert b.state(100, "CodexBot")["matching_games"] == 1
        assert len(b.analytics()["bots"]["CodexBot"]["recent"]) == 1
        for i in range(18, 26):
            b.finish(game(i, "A", "B"))
        b._compact(force=True)  # Fold the reset event itself into state.json.
        c = arena.Arena(directory, {}, False, keep=10, history=5)
        assert c.skills == b.skills and c.stats == b.stats
        assert c.base["resets"]["CodexBot"] == 16
        assert c.stats["CodexBot"]["games"] == 1
        assert c.state(100, "")["total_games"] == total + 9


if __name__ == "__main__":
    test_reset()
    print("ok: reset guards, opponent preservation, history, charts, restart, compaction")
