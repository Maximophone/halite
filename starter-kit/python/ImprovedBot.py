"""
A sparring partner with the classic beginner improvements:
  1. Capture a neighbouring site when strong enough, preferring the best
     production-to-strength ratio (and enemy sites, for overkill damage).
  2. Otherwise stay still to grow until strength >= 5x production.
  3. Interior pieces move toward the nearest border instead of idling.
Good enough to beat RandomBot every time; beatable with some thought.
"""

from hlt import *
from networking import *

myID, gameMap = getInit()
sendInit("ImprovedBot")


def site_value(site):
    if site.owner == 0 and site.strength > 0:
        return site.production / site.strength
    # Enemy or empty neutral site: cheap to take, and fighting on an enemy
    # border damages every adjacent enemy piece at once.
    return site.production + 1


def nearest_border_direction(loc):
    best_dir, best_dist = NORTH, max(gameMap.width, gameMap.height) // 2
    for d in CARDINALS:
        dist = 0
        cur = loc
        while gameMap.getSite(cur).owner == myID and dist < max(gameMap.width, gameMap.height) // 2:
            dist += 1
            cur = gameMap.getLocation(cur, d)
        if dist < best_dist:
            best_dir, best_dist = d, dist
    return best_dir


def move_for(loc):
    site = gameMap.getSite(loc)
    targets = []
    border = False
    for d in CARDINALS:
        n = gameMap.getSite(loc, d)
        if n.owner != myID:
            border = True
            if n.strength < site.strength:
                targets.append((site_value(n) + (1.0 if n.owner != 0 else 0.0), d))
    if targets:
        return Move(loc, max(targets)[1])
    if site.strength < site.production * 5:
        return Move(loc, STILL)
    if not border:
        return Move(loc, nearest_border_direction(loc))
    return Move(loc, STILL)


while True:
    gameMap = getFrame()
    moves = []
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            loc = Location(x, y)
            if gameMap.getSite(loc).owner == myID:
                moves.append(move_for(loc))
    sendFrame(moves)
