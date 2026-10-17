"""
PredatorBot: an aggressive, combat-first bot.

Concept: grow just enough economy to feed the war, find the enemy, and pour
every spare unit of strength into the front line.

  1. Hunt. A distance field from every enemy site tells each neutral site how
     close it is to prey. Neutral sites on the way to an enemy are worth more,
     so the territory grows toward the enemy instead of away from it.
  2. Overkill. A piece deals its full strength to every enemy piece on its
     site and the 4 neighbours. Combat targets are scored by the total enemy
     strength they touch, so pieces dive into gaps between several enemy
     pieces and hit them all at once.
  3. Supply lines. A max-value field (Dijkstra with decay) is propagated
     inward through our territory from the frontier. Interior pieces follow
     it, so strength streams toward the best target, and combat targets
     outrank everything else.
  4. Pack hunting. A neutral wall too strong for one piece is broken by
     several adjacent pieces moving in on the same turn.
  5. No waste. Moves that would push a site well past the 255 cap are
     cancelled, smallest contributor first.
"""

import heapq

from hlt import *
from networking import *

myID, gameMap = getInit()
sendInit("PredatorBot")

W, H = gameMap.width, gameMap.height
N = W * H
PROD = [gameMap.contents[i // W][i % W].production for i in range(N)]
# NB[i][d - 1] is the neighbour of site i in direction d (NORTH, EAST, SOUTH, WEST).
NB = []
for i in range(N):
    x, y = i % W, i // W
    NB.append((((y - 1) % H) * W + x, y * W + (x + 1) % W,
               ((y + 1) % H) * W + x, y * W + (x - 1) % W))
OPPOSITE = {NORTH: SOUTH, SOUTH: NORTH, EAST: WEST, WEST: EAST}

WAIT = 5              # interior pieces grow until strength >= WAIT * production
WAIT_FRONT = 2        # ...but only this long when feeding a fight
MIN_MOVE = 8          # never move a tiny piece across our own territory
DECAY = 0.82          # value lost per step of travel through our territory
FIGHT_VALUE = 1.2     # base value of a fight; a rich neutral site is worth about 0.5
HUNT_RANGE = 5        # neutral sites this close to an enemy get a hunting bonus
CAP_SLACK = 30        # tolerated overflow past 255 on one site


def enemy_distance(owner):
    """BFS distance from the nearest enemy site, over the whole map."""
    dist = [-1] * N
    frontier = [i for i in range(N) if owner[i] not in (0, myID)]
    for i in frontier:
        dist[i] = 0
    d = 0
    while frontier:
        d += 1
        nxt = []
        for i in frontier:
            for j in NB[i]:
                if dist[j] < 0:
                    dist[j] = d
                    nxt.append(j)
        frontier = nxt
    return dist


def turn(owner, strength):
    is_enemy = [o != 0 and o != myID for o in owner]
    # Damage an enemy presence around a site can absorb or deal: the sum of
    # enemy strength on the site and its 4 neighbours.
    heat = [0] * N
    for i in range(N):
        if is_enemy[i]:
            s = strength[i]
            heat[i] += s
            for j in NB[i]:
                heat[j] += s
    edist = enemy_distance(owner)
    at_war = any(is_enemy)

    def is_fight(t):
        return heat[t] > 0 and (is_enemy[t] or strength[t] == 0)

    def value(t):
        if is_fight(t):
            # Fight: every enemy unit in reach gets hit by the full strength.
            return FIGHT_VALUE + heat[t] / 255.0 + PROD[t] / 15.0
        # A site is also worth the ground it opens up.
        p = PROD[t] + 0.25 * sum(PROD[j] for j in NB[t] if owner[j] != myID)
        s = strength[t]
        v = p / s if s > 0 else p + 1.0
        if at_war and 0 <= edist[t] <= HUNT_RANGE:
            v *= 1.0 + 0.5 * (HUNT_RANGE - edist[t]) / HUNT_RANGE
        return v

    mine = [i for i in range(N) if owner[i] == myID]

    # Max-value field through our territory, seeded from the frontier.
    field = [-1.0] * N
    fdir = [STILL] * N
    ffight = [False] * N  # the best target for this piece is a fight
    tval = {}
    heap = []
    for i in mine:
        for d in CARDINALS:
            t = NB[i][d - 1]
            if owner[t] != myID:
                if t not in tval:
                    tval[t] = value(t)
                heap.append((-tval[t], i, d, is_fight(t)))
    heapq.heapify(heap)
    while heap:
        negv, i, d, fight = heapq.heappop(heap)
        if field[i] >= 0:
            continue
        field[i] = -negv
        fdir[i] = d
        ffight[i] = fight
        nv = -negv * DECAY
        for dd in CARDINALS:
            j = NB[i][dd - 1]
            if owner[j] == myID and field[j] < 0:
                heapq.heappush(heap, (-nv, j, OPPOSITE[dd], fight))

    move = {}
    pack = {}  # neutral target -> pieces that want to break it together
    for i in sorted(mine, key=lambda k: -strength[k]):
        s = strength[i]
        if s == 0:
            move[i] = STILL
            continue
        adjacent = [(tval[t], d, t) for d, t in zip(CARDINALS, NB[i]) if owner[t] != myID]
        if adjacent:
            # Border piece: deal with what is next to us, never turn back.
            fights = [a for a in adjacent if is_fight(a[2])]
            takes = [a for a in adjacent if not is_fight(a[2]) and s > strength[a[2]]]
            if fights:
                best = max(fights)
                ready = s >= max(PROD[i] * WAIT_FRONT, MIN_MOVE) or s > strength[best[2]]
                move[i] = best[1] if ready else STILL
            elif takes:
                move[i] = max(takes)[1]
            elif (owner[NB[i][fdir[i] - 1]] == myID and field[i] > 1.5 * max(adjacent)[0]
                  and s >= max(PROD[i] * WAIT, MIN_MOVE)):
                # Stuck facing poor ground: regroup toward a better frontier.
                move[i] = fdir[i]
            else:
                move[i] = STILL
                pack.setdefault(max(adjacent)[2], []).append(i)
        else:
            d = fdir[i]
            # Best target is further away: travel when grown enough.
            wait = WAIT_FRONT if ffight[i] else WAIT
            move[i] = d if s >= max(PROD[i] * wait, MIN_MOVE) else STILL

    # Pack hunting: break a strong neutral wall with several pieces at once.
    for t, members in pack.items():
        helpers = set(members)
        for j in NB[t]:
            if owner[j] == myID and move.get(j) == STILL and strength[j] > 0:
                helpers.add(j)
        if len(helpers) > 1 and sum(strength[j] for j in helpers) > strength[t]:
            for j in helpers:
                for d in CARDINALS:
                    if NB[j][d - 1] == t:
                        move[j] = d

    # Cap control: cancel the smallest arrivals at overflowing sites.
    for _ in range(3):
        load = {}
        arrivals = {}
        for i in mine:
            d = move[i]
            dest = i if d == STILL else NB[i][d - 1]
            load[dest] = load.get(dest, 0) + strength[i] + (PROD[i] if d == STILL else 0)
            if d != STILL:
                arrivals.setdefault(dest, []).append(i)
        changed = False
        for dest, total in list(load.items()):
            if total <= 255 + CAP_SLACK or dest not in arrivals:
                continue
            for i in sorted(arrivals[dest], key=lambda k: strength[k]):
                if total <= 255 + CAP_SLACK:
                    break
                if owner[dest] != myID and len(arrivals[dest]) == 1:
                    break  # a lone attacker keeps attacking
                # Detour through another neighbour with room, else wait.
                move[i] = STILL
                for alt in sorted(CARDINALS, key=lambda dd: -field[NB[i][dd - 1]]):
                    j = NB[i][alt - 1]
                    if (j != dest and owner[j] == myID and field[j] >= 0.8 * field[dest]
                            and load.get(j, 0) + strength[i] <= 255):
                        move[i] = alt
                        load[j] = load.get(j, 0) + strength[i]
                        break
                total -= strength[i]
                changed = True
        if not changed:
            break

    return [Move(Location(i % W, i // W), move[i]) for i in mine]


while True:
    gameMap = getFrame()
    owner = [0] * N
    strength = [0] * N
    k = 0
    for row in gameMap.contents:
        for site in row:
            owner[k] = site.owner
            strength[k] = site.strength
            k += 1
    sendFrame(turn(owner, strength))
