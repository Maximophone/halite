//! PredatorBot: an aggressive, combat-first bot. A faithful port of
//! starter-kit/python/PredatorBot.py (same constants, same logic, same
//! tie-breaking), so both versions pick the same moves on the same map.
//!
//!   1. Hunt. Neutral sites on the way to an enemy are worth more.
//!   2. Overkill. Combat targets are scored by the total enemy strength they touch.
//!   3. Supply lines. A max-value field (Dijkstra with decay) through our
//!      territory pulls interior strength toward the best target.
//!   4. Pack hunting. Several adjacent pieces break a strong neutral wall together.
//!   5. No waste. Moves that push a site well past the 255 cap are cancelled.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use hlt::*;

const WAIT: i32 = 5; // interior pieces grow until strength >= WAIT * production
const WAIT_FRONT: i32 = 2; // ...but only this long when feeding a fight
const MIN_MOVE: i32 = 8; // never move a tiny piece across our own territory
const DECAY: f64 = 0.82; // value lost per step of travel through our territory
const FIGHT_VALUE: f64 = 1.2; // base value of a fight; a rich neutral site is worth about 0.5
const HUNT_RANGE: i32 = 5; // neutral sites this close to an enemy get a hunting bonus
const CAP_SLACK: i32 = 30; // tolerated overflow past 255 on one site

fn opposite(d: u8) -> u8 {
    match d {
        NORTH => SOUTH,
        SOUTH => NORTH,
        EAST => WEST,
        _ => EAST,
    }
}

/// Heap entry ordered like the Python tuple (-value, site, direction, fight),
/// popped smallest first.
struct Entry {
    negv: f64,
    i: usize,
    d: u8,
    fight: bool,
}

impl Entry {
    fn key_cmp(&self, o: &Entry) -> Ordering {
        self.negv
            .partial_cmp(&o.negv)
            .unwrap()
            .then(self.i.cmp(&o.i))
            .then(self.d.cmp(&o.d))
            .then(self.fight.cmp(&o.fight))
    }
}

impl PartialEq for Entry {
    fn eq(&self, o: &Entry) -> bool {
        self.key_cmp(o) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Entry) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Entry {
    fn cmp(&self, o: &Entry) -> Ordering {
        o.key_cmp(self) // reversed: BinaryHeap is a max-heap
    }
}

/// Python's max() over (value, direction, site) tuples.
fn tuple_max(v: &[(f64, u8, usize)]) -> (f64, u8, usize) {
    let mut best = v[0];
    for &a in &v[1..] {
        let ord = a.0.partial_cmp(&best.0).unwrap().then(a.1.cmp(&best.1)).then(a.2.cmp(&best.2));
        if ord == Ordering::Greater {
            best = a;
        }
    }
    best
}

struct Bot {
    my_id: i32,
    w: usize,
    n: usize,
    prod: Vec<i32>,
    /// nb[i][d - 1] is the neighbour of site i in direction d (NORTH, EAST, SOUTH, WEST).
    nb: Vec<[usize; 4]>,
}

impl Bot {
    /// BFS distance from the nearest enemy site, over the whole map.
    fn enemy_distance(&self, owner: &[i32]) -> Vec<i32> {
        let mut dist = vec![-1; self.n];
        let mut frontier: Vec<usize> =
            (0..self.n).filter(|&i| owner[i] != 0 && owner[i] != self.my_id).collect();
        for &i in &frontier {
            dist[i] = 0;
        }
        let mut d = 0;
        while !frontier.is_empty() {
            d += 1;
            let mut nxt = Vec::new();
            for &i in &frontier {
                for &j in &self.nb[i] {
                    if dist[j] < 0 {
                        dist[j] = d;
                        nxt.push(j);
                    }
                }
            }
            frontier = nxt;
        }
        dist
    }

    fn turn(&self, owner: &[i32], strength: &[i32]) -> Vec<Move> {
        let (n, my_id, prod, nb) = (self.n, self.my_id, &self.prod, &self.nb);
        let is_enemy: Vec<bool> = owner.iter().map(|&o| o != 0 && o != my_id).collect();
        // Damage an enemy presence around a site can absorb or deal.
        let mut heat = vec![0i32; n];
        for i in 0..n {
            if is_enemy[i] {
                let s = strength[i];
                heat[i] += s;
                for &j in &nb[i] {
                    heat[j] += s;
                }
            }
        }
        let edist = self.enemy_distance(owner);
        let at_war = is_enemy.iter().any(|&e| e);

        let is_fight = |t: usize| heat[t] > 0 && (is_enemy[t] || strength[t] == 0);

        let value = |t: usize| -> f64 {
            if is_fight(t) {
                // Fight: every enemy unit in reach gets hit by the full strength.
                return FIGHT_VALUE + heat[t] as f64 / 255.0 + prod[t] as f64 / 15.0;
            }
            // A site is also worth the ground it opens up.
            let around: i32 = nb[t].iter().filter(|&&j| owner[j] != my_id).map(|&j| prod[j]).sum();
            let p = prod[t] as f64 + 0.25 * around as f64;
            let s = strength[t];
            let mut v = if s > 0 { p / s as f64 } else { p + 1.0 };
            if at_war && 0 <= edist[t] && edist[t] <= HUNT_RANGE {
                v *= 1.0 + 0.5 * (HUNT_RANGE - edist[t]) as f64 / HUNT_RANGE as f64;
            }
            v
        };

        let mine: Vec<usize> = (0..n).filter(|&i| owner[i] == my_id).collect();

        // Max-value field through our territory, seeded from the frontier.
        let mut field = vec![-1.0f64; n];
        let mut fdir = vec![STILL; n];
        let mut ffight = vec![false; n]; // the best target for this piece is a fight
        let mut tval = vec![f64::NAN; n];
        let mut heap = BinaryHeap::new();
        for &i in &mine {
            for d in CARDINALS {
                let t = nb[i][d as usize - 1];
                if owner[t] != my_id {
                    if tval[t].is_nan() {
                        tval[t] = value(t);
                    }
                    heap.push(Entry { negv: -tval[t], i, d, fight: is_fight(t) });
                }
            }
        }
        while let Some(Entry { negv, i, d, fight }) = heap.pop() {
            if field[i] >= 0.0 {
                continue;
            }
            field[i] = -negv;
            fdir[i] = d;
            ffight[i] = fight;
            let nv = -negv * DECAY;
            for dd in CARDINALS {
                let j = nb[i][dd as usize - 1];
                if owner[j] == my_id && field[j] < 0.0 {
                    heap.push(Entry { negv: -nv, i: j, d: opposite(dd), fight });
                }
            }
        }

        let mut mv = vec![STILL; n];
        // Neutral target -> pieces that want to break it together, in insertion order.
        let mut pack: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut by_strength = mine.clone();
        by_strength.sort_by_key(|&k| -strength[k]); // stable, like Python's sorted
        for &i in &by_strength {
            let s = strength[i];
            if s == 0 {
                mv[i] = STILL;
                continue;
            }
            let adjacent: Vec<(f64, u8, usize)> = CARDINALS
                .iter()
                .zip(nb[i].iter())
                .filter(|(_, &t)| owner[t] != my_id)
                .map(|(&d, &t)| (tval[t], d, t))
                .collect();
            if !adjacent.is_empty() {
                // Border piece: deal with what is next to us, never turn back.
                let fights: Vec<_> = adjacent.iter().copied().filter(|a| is_fight(a.2)).collect();
                let takes: Vec<_> = adjacent
                    .iter()
                    .copied()
                    .filter(|a| !is_fight(a.2) && s > strength[a.2])
                    .collect();
                let best_adj = tuple_max(&adjacent);
                if !fights.is_empty() {
                    let best = tuple_max(&fights);
                    let ready = s >= (prod[i] * WAIT_FRONT).max(MIN_MOVE) || s > strength[best.2];
                    mv[i] = if ready { best.1 } else { STILL };
                } else if !takes.is_empty() {
                    mv[i] = tuple_max(&takes).1;
                } else if owner[nb[i][fdir[i] as usize - 1]] == my_id
                    && field[i] > 1.5 * best_adj.0
                    && s >= (prod[i] * WAIT).max(MIN_MOVE)
                {
                    // Stuck facing poor ground: regroup toward a better frontier.
                    mv[i] = fdir[i];
                } else {
                    mv[i] = STILL;
                    match pack.iter_mut().find(|(t, _)| *t == best_adj.2) {
                        Some((_, members)) => members.push(i),
                        None => pack.push((best_adj.2, vec![i])),
                    }
                }
            } else {
                let d = fdir[i];
                // Best target is further away: travel when grown enough.
                let wait = if ffight[i] { WAIT_FRONT } else { WAIT };
                mv[i] = if s >= (prod[i] * wait).max(MIN_MOVE) { d } else { STILL };
            }
        }

        // Pack hunting: break a strong neutral wall with several pieces at once.
        for (t, members) in &pack {
            let mut helpers = members.clone();
            for &j in &nb[*t] {
                if owner[j] == my_id && mv[j] == STILL && strength[j] > 0 && !helpers.contains(&j) {
                    helpers.push(j);
                }
            }
            if helpers.len() > 1 && helpers.iter().map(|&j| strength[j]).sum::<i32>() > strength[*t] {
                for &j in &helpers {
                    for d in CARDINALS {
                        if nb[j][d as usize - 1] == *t {
                            mv[j] = d;
                        }
                    }
                }
            }
        }

        // Cap control: cancel the smallest arrivals at overflowing sites.
        let dest_of = |i: usize, d: u8| if d == STILL { i } else { nb[i][d as usize - 1] };
        for _ in 0..3 {
            let mut load = vec![0i32; n];
            let mut arrivals: Vec<Vec<usize>> = vec![Vec::new(); n];
            let mut seen = vec![false; n];
            let mut order = Vec::new(); // destinations in first-seen order, like a dict
            for &i in &mine {
                let d = mv[i];
                let dest = dest_of(i, d);
                if !seen[dest] {
                    seen[dest] = true;
                    order.push(dest);
                }
                load[dest] += strength[i] + if d == STILL { prod[i] } else { 0 };
                if d != STILL {
                    arrivals[dest].push(i);
                }
            }
            let snapshot: Vec<(usize, i32)> = order.iter().map(|&k| (k, load[k])).collect();
            let mut changed = false;
            for &(dest, total0) in &snapshot {
                let mut total = total0;
                if total <= 255 + CAP_SLACK || arrivals[dest].is_empty() {
                    continue;
                }
                let mut arr = arrivals[dest].clone();
                arr.sort_by_key(|&k| strength[k]);
                for &i in &arr {
                    if total <= 255 + CAP_SLACK {
                        break;
                    }
                    if owner[dest] != my_id && arrivals[dest].len() == 1 {
                        break; // a lone attacker keeps attacking
                    }
                    // Detour through another neighbour with room, else wait.
                    mv[i] = STILL;
                    let mut alts = CARDINALS;
                    alts.sort_by(|&a, &b| {
                        field[nb[i][b as usize - 1]].partial_cmp(&field[nb[i][a as usize - 1]]).unwrap()
                    });
                    for alt in alts {
                        let j = nb[i][alt as usize - 1];
                        if j != dest
                            && owner[j] == my_id
                            && field[j] >= 0.8 * field[dest]
                            && load[j] + strength[i] <= 255
                        {
                            mv[i] = alt;
                            load[j] += strength[i];
                            break;
                        }
                    }
                    total -= strength[i];
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        mine.iter()
            .map(|&i| Move {
                loc: Location { x: (i % self.w) as u16, y: (i / self.w) as u16 },
                direction: mv[i],
            })
            .collect()
    }
}

fn main() {
    let (my_id, game_map, mut conn) = get_init();
    send_init("RustPredatorBot");

    let (w, h) = (game_map.width as usize, game_map.height as usize);
    let n = w * h;
    let prod: Vec<i32> = (0..n).map(|i| game_map.contents[i / w][i % w].production as i32).collect();
    let nb: Vec<[usize; 4]> = (0..n)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            [((y + h - 1) % h) * w + x, y * w + (x + 1) % w, ((y + 1) % h) * w + x, y * w + (x + w - 1) % w]
        })
        .collect();
    let bot = Bot { my_id: my_id as i32, w, n, prod, nb };

    let mut owner = vec![0i32; n];
    let mut strength = vec![0i32; n];
    loop {
        let game_map = conn.get_frame();
        for (k, site) in game_map.contents.iter().flatten().enumerate() {
            owner[k] = site.owner as i32;
            strength[k] = site.strength as i32;
        }
        let moves = bot.turn(&owner, &strength);
        conn.send_frame(&moves);
    }
}
