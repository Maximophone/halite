"""Halite I networking (same API as the original 2016 Python starter kit).

Talks to the game environment over stdin/stdout. Never print() to stdout
from your bot: use a log file or sys.stderr (the environment saves stderr
to logs/bot-N.log).
"""

import sys
from hlt import GameMap, Site

_productions = []
_width = -1
_height = -1


def serializeMoveSet(moves):
    return " ".join("%d %d %d" % (m.loc.x, m.loc.y, m.direction) for m in moves)


def deserializeMapSize(inputString):
    global _width, _height
    split = inputString.split()
    _width = int(split[0])
    _height = int(split[1])


def deserializeProductions(inputString):
    global _productions
    split = inputString.split()
    _productions = []
    i = 0
    for a in range(0, _height):
        row = []
        for b in range(0, _width):
            row.append(int(split[i]))
            i += 1
        _productions.append(row)


def deserializeMap(inputString):
    split = inputString.split()
    m = GameMap(_width, _height)

    y = 0
    x = 0
    counter = 0
    owner = 0
    while y != m.height:
        counter = int(split.pop(0))
        owner = int(split.pop(0))
        for a in range(0, counter):
            m.contents[y][x].owner = owner
            x += 1
            if x == m.width:
                x = 0
                y += 1

    split = split[::-1]  # pop from the end for speed
    for a in range(0, _height):
        for b in range(0, _width):
            m.contents[a][b].strength = int(split.pop())
            m.contents[a][b].production = _productions[a][b]

    return m


def sendString(toBeSent):
    sys.stdout.write(toBeSent.replace("\n", "") + "\n")
    sys.stdout.flush()


def getString():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)  # environment closed the pipe: game over
    return line.rstrip("\n")


def getInit():
    playerTag = int(getString())
    deserializeMapSize(getString())
    deserializeProductions(getString())
    m = deserializeMap(getString())
    return (playerTag, m)


def sendInit(name):
    sendString(name)


def getFrame():
    return deserializeMap(getString())


def sendFrame(moves):
    sendString(serializeMoveSet(moves))
