#!/usr/bin/env python3
"""Validate the release policy against every built, registered opponent."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", default="starter-kit/rust/target/codex-final/release/codex_bot")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--erdman-games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="starter-kit/rust/target/codex-results/validation")
    a = ap.parse_args()
    opponents = [p.stem for p in (ROOT / "starter-kit/rust/src/bin").glob("*.rs")
                 if p.stem != "codex_bot" and (ROOT / "starter-kit/rust/target/release" / p.stem).is_file()]
    opponents.sort(key=lambda name: (name != "erdman_bot", name))
    destination = ROOT / a.out
    destination.mkdir(parents=True, exist_ok=True)
    summary = {}
    for opponent in opponents:
        print(f"Validating against {opponent}", flush=True)
        result_file = destination / f"{opponent}.json"
        subprocess.run([sys.executable, str(Path(__file__).with_name("sweep.py")),
                        str(Path(__file__).with_name("release.json")),
                        "--candidate", a.candidate, "--opponents", opponent,
                        "--games", str(a.erdman_games if opponent == "erdman_bot" else a.games),
                        "--seed", str(a.seed), "--workers", str(a.workers),
                        "--out", str(result_file)], cwd=ROOT, check=True)
        result = json.loads(result_file.read_text())
        r = result["experiments"]["release"]
        summary[opponent] = {"wins": r["wins"], "games": len(r["games"]),
                             "errors": r["errors"], "opponent_errors": r["opponent_errors"],
                             "sha256": result["sha256"]}
        (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
