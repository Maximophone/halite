#!/usr/bin/env python3
"""
Live training dashboard for the rl/ trainer (Python standard library only).

    python3 rl/dashboard.py                 # http://localhost:8090
    python3 rl/dashboard.py --port 8091 --runs rl/runs

Serves rl/dashboard.html and /api/runs, which parses every <runs>/<name>/log.txt written by
`halite-rl train` (train lines every 100 steps, eval lines every --eval-every steps).
"""

import argparse
import json
import os
import re
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
STEP = re.compile(r"^step\s+(\d+)\s+loss ([\d.]+)\s+acc ([\d.]+)\s+lr ([\d.e+-]+)\s+(\d+) steps/min(?:\s+data wait (\d+)%)?")
EVAL = re.compile(r"^eval step (\d+)\s+val loss ([\d.]+)\s+acc ([\d.]+)\s+move recall ([\d.]+)")
INFO = re.compile(r"^(\d+) games, (\d+) samples")
MODEL = re.compile(r"^model: (\d+) channels, dilations \[([\d, ]+)\], (\d+) parameters")


def parse_run(path):
    run = {"steps": [], "evals": [], "total": None, "games": None, "samples": None, "params": None, "model": None}
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("args:"):
                # a restarted run appends to the same log: keep only the latest session,
                # unless it resumes (--resume STEP), which continues the same curves
                r = re.search(r'"--resume", "(\d+)"', line)
                if r:  # drop what the resumed run will redo
                    run["steps"] = [x for x in run["steps"] if x["step"] <= int(r.group(1))]
                    run["evals"] = [x for x in run["evals"] if x["step"] <= int(r.group(1))]
                else:
                    run["steps"], run["evals"] = [], []
                m = re.search(r'"--steps", "(\d+)"', line)
                run["total"] = int(m.group(1)) if m else 20000
                run["args"] = line[5:].strip()
            elif m := STEP.match(line):
                run["steps"].append({"step": int(m[1]), "loss": float(m[2]), "acc": float(m[3]), "lr": float(m[4]),
                                     "spm": int(m[5]), "wait": int(m[6] or 0)})
            elif m := EVAL.match(line):
                run["evals"].append({"step": int(m[1]), "loss": float(m[2]), "acc": float(m[3]), "recall": float(m[4])})
            elif m := INFO.match(line):
                run["games"], run["samples"] = int(m[1]), int(m[2])
            elif m := MODEL.match(line):
                run["params"] = int(m[3])
                run["model"] = "%s ch, dilations %s" % (m[1], m[2])
    run["updated"] = os.path.getmtime(path)
    return run


class Handler(SimpleHTTPRequestHandler):
    runs_dir = "rl/runs"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            body = open(os.path.join(HERE, "dashboard.html"), "rb").read()
            return self.reply(body, "text/html; charset=utf-8")
        if self.path.startswith("/api/runs"):
            runs = {}
            if os.path.isdir(self.runs_dir):
                for name in sorted(os.listdir(self.runs_dir)):
                    p = os.path.join(self.runs_dir, name, "log.txt")
                    if os.path.isfile(p):
                        runs[name] = parse_run(p)
            body = json.dumps({"now": time.time(), "runs": runs}).encode()
            return self.reply(body, "application/json")
        self.send_error(404)

    def reply(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    a = ap.parse_args()
    Handler.runs_dir = a.runs
    print("training dashboard on http://localhost:%d (runs in %s)" % (a.port, a.runs), flush=True)
    HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
