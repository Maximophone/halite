//! ApexAlpha: first prototype of the Apex line. Expand hard, never start a
//! fight with a strong neighbour, win the fights that come.
//!
//!   1. Walls. A neutral site touching enemy land is a wall: we leave it
//!      standing, so neighbours waste strength breaking in while we grow.
//!   2. Combat micro. Near a fight, every piece weighs staying against moving
//!      into each enemy or empty site by damage dealt minus damage taken,
//!      tracking what each enemy piece has left, so strength focuses where it
//!      kills instead of piling onto one target (overkill done right).
//!   3. Regional value. A neutral site is worth its production per unit of
//!      strength, plus part of the best ground it opens up.
//!   4. Supply lines. A decaying max-value field through our territory routes
//!      interior strength to the best frontier; fights pull harder.
//!   5. Gathering. For each frontier target, the turns needed to bring enough
//!      strength from pieces up to 8 steps away; targets ranked by value per
//!      turn of delay (to the power 0.25), pieces reserved, the plan redone
//!      every turn. About 20% faster expansion than PredatorBot alone.
//!   6. Defence. Pieces within 5 steps of an enemy are kept out of expansion,
//!      and our sites next to an enemy pull interior strength.
//!   7. Packs and caps. Adjacent pieces break strong sites together; moves
//!      that would waste strength past 255 detour or wait.
//!
//! Measured against PredatorBot (100 games each): 88% of 1v1s, 61% of
//! 4-player games against two PredatorBots and an ImprovedBot.
//!
//! Every constant can be overridden with an environment variable (APEX_WAIT,
//! APEX_DECAY, ...), which is how they were tuned.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use hlt::*;

struct Params {
    wait: f64,        // interior pieces grow until strength >= wait * production
    wait_front: f64,  // ...or this when the field leads to a fight
    min_move: i32,    // never move a tiny piece across our own territory
    decay: f64,       // field value kept per step of travel
    fight_decay: f64, // ...for a fight: lower, so only nearby pieces answer it
    fight_base: f64,  // field value of a fight (rich neutral sites are about 0.3-0.6)
    open_weight: f64, // share of the best neighbouring site added to a site's value
    wall_value: f64,  // multiplier on a wall's value (0 = never break walls)
    harm_weight: f64, // damage taken counts this much against damage dealt
    cap_slack: i32,   // tolerated overflow past 255 on one site
    regroup: f64,     // a stuck border piece regroups if the field is this much better
    region: f64,      // weight of the surrounding production (blurred map) in a site's value
    prey_ratio: f64,  // break walls into players whose total strength is below this share of ours
    hunt: f64,        // extra value (as a multiplier) for sites touching a weak player's land
    gather: f64,      // gathering search depth in steps (0 = off)
    delay_pow: f64,   // gathering ranks targets by value / turns ^ delay_pow
    danger: f64,      // pieces this close to an enemy are kept for defence, not expansion (-1 = off)
    front: f64,       // our sites this close to an enemy pull interior strength...
    defend: f64,      // ...with this field value (0 = off)
    min_cost: f64,    // value = production / max(strength, min_cost): free sites need no
                      // strength sent to them, so they must not outshine everything
}

fn env(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

impl Params {
    fn load() -> Params {
        Params {
            wait: env("APEX_WAIT", 5.0),
            wait_front: env("APEX_WAIT_FRONT", 2.0),
            min_move: env("APEX_MIN_MOVE", 25.0) as i32,
            decay: env("APEX_DECAY", 0.8),
            fight_decay: env("APEX_FIGHT_DECAY", 0.55),
            fight_base: env("APEX_FIGHT", 1.5),
            open_weight: env("APEX_OPEN", 0.5),
            wall_value: env("APEX_WALL", 0.0),
            harm_weight: env("APEX_HARM", 0.6),
            cap_slack: env("APEX_CAP", 20.0) as i32,
            regroup: env("APEX_REGROUP", 1.5),
            region: env("APEX_REGION", 0.3),
            min_cost: env("APEX_MIN_COST", 15.0),
            prey_ratio: env("APEX_PREY", 0.6),
            hunt: env("APEX_HUNT", 1.0),
            gather: env("APEX_GATHER", 8.0),
            delay_pow: env("APEX_DELAY_POW", 0.25),
            danger: env("APEX_DANGER", 5.0),
            front: env("APEX_FRONT", 2.0),
            defend: env("APEX_DEFEND", 0.6),
        }
    }
}

fn opposite(d: u8) -> u8 {
    match d {
        NORTH => SOUTH,
        SOUTH => NORTH,
        EAST => WEST,
        _ => EAST,
    }
}

struct Entry {
    v: f64,
    i: usize,
    d: u8,
    fight: bool,
}
impl PartialEq for Entry {
    fn eq(&self, o: &Entry) -> bool {
        self.cmp(o) == Ordering::Equal
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
        // Max-heap on value; ties broken by site then direction for determinism.
        self.v.partial_cmp(&o.v).unwrap().then(o.i.cmp(&self.i)).then(o.d.cmp(&self.d))
    }
}

struct Bot {
    me: i32,
    w: usize,
    n: usize,
    prod: Vec<i32>,
    /// Production of the surrounding area (radius 4, weighted by 1 / (1 + distance)),
    /// so that a site in a desert still points toward rich ground.
    region: Vec<f64>,
    nb: Vec<[usize; 4]>,
    p: Params,
}

impl Bot {
    fn turn(&self, owner: &[i32], strength: &[i32]) -> Vec<Move> {
        let (n, me, prod, nb, p) = (self.n, self.me, &self.prod, &self.nb, &self.p);
        let mine_at = |i: usize| owner[i] == me;
        let enemy_at = |i: usize| owner[i] != 0 && owner[i] != me;

        // Enemy strength that can hit a piece standing on each site (enemies staying put).
        let mut heat = vec![0i32; n];
        for i in 0..n {
            if enemy_at(i) && strength[i] > 0 {
                heat[i] += strength[i];
                for &j in &nb[i] {
                    heat[j] += strength[i];
                }
            }
        }
        // Total strength per player: weak neighbours are prey, strong ones are left alone.
        let players = owner.iter().copied().max().unwrap_or(0).max(me) as usize;
        let mut total = vec![0i64; players + 1];
        for i in 0..n {
            total[owner[i] as usize] += strength[i] as i64;
        }
        let prey = |q: i32| (total[q as usize] as f64) < p.prey_ratio * total[me as usize] as f64;
        // A wall only protects us from players we do not want to fight.
        let touches_enemy = |t: usize| nb[t].iter().any(|&j| enemy_at(j) && !prey(owner[j]));
        // A fight site: enemy land, or an empty neutral gap with enemies in reach.
        let is_fight = |t: usize| !mine_at(t) && (enemy_at(t) || (strength[t] == 0 && heat[t] > 0));
        let is_wall = |t: usize| owner[t] == 0 && strength[t] > 0 && touches_enemy(t);

        let base = |t: usize| -> f64 {
            let s = strength[t];
            let pr = prod[t] as f64 + p.region * self.region[t];
            pr / (s as f64).max(p.min_cost)
        };
        let value = |t: usize| -> f64 {
            if is_fight(t) {
                return p.fight_base + heat[t] as f64 / 255.0;
            }
            let mut open: f64 = 0.0;
            for &j in &nb[t] {
                if owner[j] == 0 && !is_wall(j) {
                    open = open.max(base(j));
                }
            }
            let mut v = base(t) + p.open_weight * open;
            if nb[t].iter().any(|&j| enemy_at(j) && prey(owner[j])) {
                v *= 1.0 + p.hunt;
            }
            if is_wall(t) { v * p.wall_value } else { v }
        };

        let mine: Vec<usize> = (0..n).filter(|&i| mine_at(i)).collect();

        // Steps to the nearest enemy site.
        let mut edist = vec![i32::MAX; n];
        let mut layer: Vec<usize> = (0..n).filter(|&i| enemy_at(i)).collect();
        for &i in &layer {
            edist[i] = 0;
        }
        let mut dd = 0;
        while !layer.is_empty() {
            dd += 1;
            let mut next = Vec::new();
            for &i in &layer {
                for &j in &nb[i] {
                    if edist[j] == i32::MAX {
                        edist[j] = dd;
                        next.push(j);
                    }
                }
            }
            layer = next;
        }
        let in_danger = |i: usize| (edist[i] as f64) <= p.danger;

        // Max-value field through our territory, seeded from frontier targets.
        let mut field = vec![-1.0f64; n];
        let mut fdir = vec![STILL; n];
        let mut ffight = vec![false; n];
        let mut tval = vec![f64::NAN; n];
        let mut heap = BinaryHeap::new();
        for &i in &mine {
            for d in CARDINALS {
                let t = nb[i][d as usize - 1];
                if !mine_at(t) {
                    if tval[t].is_nan() {
                        tval[t] = value(t);
                    }
                    if tval[t] > 0.0 {
                        heap.push(Entry { v: tval[t], i, d, fight: is_fight(t) });
                    }
                }
            }
        }
        // Defence: our sites near an enemy pull strength so the front stays strong.
        if p.defend > 0.0 {
            for &i in &mine {
                if (edist[i] as f64) <= p.front {
                    heap.push(Entry { v: p.defend, i, d: STILL, fight: true });
                }
            }
        }
        while let Some(Entry { v, i, d, fight }) = heap.pop() {
            if field[i] >= 0.0 {
                continue;
            }
            field[i] = v;
            fdir[i] = d;
            ffight[i] = fight;
            for dd in CARDINALS {
                let j = nb[i][dd as usize - 1];
                if mine_at(j) && field[j] < 0.0 {
                    let k = if fight { p.fight_decay } else { p.decay };
                    heap.push(Entry { v: v * k, i: j, d: opposite(dd), fight });
                }
            }
        }

        let mut mv = vec![STILL; n];
        let mut done = vec![false; n];
        let mut by_strength = mine.clone();
        by_strength.sort_by_key(|&k| (-strength[k], k));

        // 1. Combat micro. `left` is what each enemy piece has left after the
        //    damage we already plan to deal this turn.
        let mut left: Vec<i32> = (0..n).map(|i| if enemy_at(i) { strength[i] } else { 0 }).collect();
        let dealt = |c: usize, s: i32, left: &Vec<i32>| -> i32 {
            let mut d = left[c].min(s);
            for &j in &nb[c] {
                d += left[j].min(s);
            }
            d
        };
        for &i in &by_strength {
            let s = strength[i];
            if s == 0 {
                continue;
            }
            let near_fight = heat[i] > 0 || nb[i].iter().any(|&t| is_fight(t));
            if !near_fight {
                continue;
            }
            // Staying: we hit what is around us and take what is around us.
            let mut best_d = STILL;
            let mut best_score = dealt(i, s, &left) as f64 - p.harm_weight * heat[i].min(s) as f64;
            for d in CARDINALS {
                let t = nb[i][d as usize - 1];
                if !is_fight(t) {
                    continue;
                }
                let gain = dealt(t, s, &left) as f64;
                let harm = (heat[t] + if owner[t] == 0 { strength[t] } else { 0 }).min(s) as f64;
                // Taking the site is worth a little production on top.
                let score = gain - p.harm_weight * harm + prod[t] as f64;
                if score > best_score {
                    best_score = score;
                    best_d = d;
                }
            }
            let target = if best_d == STILL { i } else { nb[i][best_d as usize - 1] };
            if best_d == STILL && dealt(i, s, &left) == 0 {
                continue; // nothing to fight from here; let expansion decide
            }
            mv[i] = best_d;
            done[i] = true;
            left[target] = (left[target] - s).max(0);
            for &j in &nb[target] {
                left[j] = (left[j] - s).max(0);
            }
        }

        // 2. Gathering. For each frontier target, find how many turns k it takes
        //    to bring enough strength from pieces up to `gather` steps away (a
        //    piece d steps away arrives in d turns; closer ones grow while they
        //    wait). Targets go in order of value per turn of delay; each reserves
        //    the pieces it needs. Pieces exactly k steps away leave now, closer
        //    ones wait, and the plan is redone every turn.
        if p.gather > 0.0 {
            let depth = p.gather as usize;
            let mut dist = vec![usize::MAX; n];
            let mut dir_to = vec![STILL; n];
            let mut touched: Vec<usize> = Vec::new();
            // BFS from target t through our territory; returns pieces by distance.
            let plan = |t: usize, done: &Vec<bool>, dist: &mut Vec<usize>, dir_to: &mut Vec<u8>,
                            touched: &mut Vec<usize>| -> Option<(usize, Vec<usize>)> {
                for &j in touched.iter() {
                    dist[j] = usize::MAX;
                }
                touched.clear();
                let mut layer: Vec<usize> = Vec::new();
                for d in CARDINALS {
                    let j = nb[t][d as usize - 1];
                    if mine_at(j) && dist[j] == usize::MAX {
                        dist[j] = 1;
                        dir_to[j] = opposite(d);
                        touched.push(j);
                        layer.push(j);
                    }
                }
                let mut members: Vec<usize> = Vec::new();
                for k in 1..=depth {
                    for &j in &layer {
                        if !done[j] && strength[j] > 0 && !in_danger(j) {
                            members.push(j);
                        }
                    }
                    let avail: i32 = members.iter().map(|&j| strength[j] + prod[j] * (k - dist[j]) as i32).sum();
                    if avail > strength[t] {
                        return Some((k, members));
                    }
                    if k == depth {
                        break;
                    }
                    let mut next = Vec::new();
                    for &j in &layer {
                        for d in CARDINALS {
                            let q = nb[j][d as usize - 1];
                            if mine_at(q) && dist[q] == usize::MAX && !done[q] {
                                dist[q] = k + 1;
                                dir_to[q] = opposite(d);
                                touched.push(q);
                                next.push(q);
                            }
                        }
                    }
                    layer = next;
                }
                None
            };
            let mut cands: Vec<(f64, usize)> = Vec::new();
            for t in 0..n {
                if mine_at(t) || tval[t].is_nan() || tval[t] <= 0.0 || is_fight(t) {
                    continue;
                }
                if let Some((k, _)) = plan(t, &done, &mut dist, &mut dir_to, &mut touched) {
                    cands.push((tval[t] / (k as f64).powf(p.delay_pow), t));
                }
            }
            cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            for &(_, t) in &cands {
                let Some((k, mut members)) = plan(t, &done, &mut dist, &mut dir_to, &mut touched) else { continue };
                // Use as few pieces as possible: nearest and strongest first.
                members.sort_by_key(|&j| (dist[j], -strength[j]));
                let mut sum = 0;
                for &j in &members {
                    if sum > strength[t] {
                        break;
                    }
                    sum += strength[j] + prod[j] * (k - dist[j]) as i32;
                    done[j] = true;
                    mv[j] = if dist[j] == k { dir_to[j] } else { STILL };
                }
            }
        }

        // 3. Border pieces: capture, regroup, or wait (and maybe pack up).
        let mut pack: Vec<(usize, Vec<usize>)> = Vec::new();
        for &i in &by_strength {
            if done[i] {
                continue;
            }
            let s = strength[i];
            if s == 0 {
                continue;
            }
            let mut best_take: Option<(f64, u8)> = None;
            let mut best_any: Option<(f64, usize)> = None;
            for d in CARDINALS {
                let t = nb[i][d as usize - 1];
                if mine_at(t) || is_fight(t) {
                    continue;
                }
                let v = tval[t];
                if v <= 0.0 {
                    continue;
                }
                if s > strength[t] && best_take.map_or(true, |(bv, _)| v > bv) {
                    best_take = Some((v, d));
                }
                if best_any.map_or(true, |(bv, _)| v > bv) {
                    best_any = Some((v, t));
                }
            }
            let border = nb[i].iter().any(|&t| !mine_at(t));
            if let Some((_, d)) = best_take {
                mv[i] = d;
                done[i] = true;
            } else if border {
                let local = best_any.map_or(0.0, |(v, _)| v);
                let fd = fdir[i];
                if fd != STILL
                    && mine_at(nb[i][fd as usize - 1])
                    && field[i] > p.regroup * local
                    && s >= ((prod[i] as f64 * p.wait) as i32).max(p.min_move)
                {
                    mv[i] = fd;
                } else if let Some((_, t)) = best_any {
                    match pack.iter_mut().find(|(pt, _)| *pt == t) {
                        Some((_, m)) => m.push(i),
                        None => pack.push((t, vec![i])),
                    }
                }
                done[i] = true;
            }
        }

        // 3. Interior pieces follow the field once grown.
        for &i in &mine {
            if done[i] || strength[i] == 0 || fdir[i] == STILL {
                continue;
            }
            let wait = if ffight[i] { p.wait_front } else { p.wait };
            if strength[i] >= ((prod[i] as f64 * wait) as i32).max(p.min_move) {
                mv[i] = fdir[i];
            }
        }

        // 4. Packs: several waiting pieces take a strong site together.
        for (t, members) in &pack {
            let mut helpers: Vec<usize> = members.clone();
            for &j in &nb[*t] {
                if mine_at(j) && mv[j] == STILL && strength[j] > 0 && !helpers.contains(&j) && !done_fight(&mv, j) {
                    helpers.push(j);
                }
            }
            let total: i32 = helpers.iter().map(|&j| strength[j]).sum();
            if helpers.len() > 1 && total > strength[*t] {
                for &j in &helpers {
                    for d in CARDINALS {
                        if nb[j][d as usize - 1] == *t {
                            mv[j] = d;
                        }
                    }
                }
            }
        }

        // 5. Cap control: cancel or detour the smallest arrivals at overflowing sites.
        for _ in 0..3 {
            let mut load = vec![0i32; n];
            let mut arrivals: Vec<Vec<usize>> = vec![Vec::new(); n];
            for &i in &mine {
                let d = mv[i];
                let dest = if d == STILL { i } else { nb[i][d as usize - 1] };
                load[dest] += strength[i] + if d == STILL { prod[i] } else { 0 };
                if d != STILL {
                    arrivals[dest].push(i);
                }
            }
            let mut changed = false;
            for dest in 0..n {
                if load[dest] <= 255 + p.cap_slack || arrivals[dest].is_empty() {
                    continue;
                }
                if !mine_at(dest) && arrivals[dest].len() == 1 {
                    continue; // a lone attacker keeps attacking
                }
                let mut arr = arrivals[dest].clone();
                arr.sort_by_key(|&k| (strength[k], k));
                for &i in &arr {
                    if load[dest] <= 255 + p.cap_slack {
                        break;
                    }
                    mv[i] = STILL;
                    let mut alts = CARDINALS;
                    alts.sort_by(|&a, &b| {
                        field[nb[i][b as usize - 1]].partial_cmp(&field[nb[i][a as usize - 1]]).unwrap()
                    });
                    for alt in alts {
                        let j = nb[i][alt as usize - 1];
                        if j != dest && mine_at(j) && field[j] >= 0.8 * field[dest] && load[j] + strength[i] <= 255 {
                            mv[i] = alt;
                            load[j] += strength[i];
                            break;
                        }
                    }
                    load[dest] -= strength[i];
                    if mv[i] == STILL {
                        load[i] += strength[i] + prod[i];
                    }
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        if std::env::var("APEX_DEBUG").is_ok() {
            let border: Vec<usize> = mine.iter().copied().filter(|&i| nb[i].iter().any(|&t| !mine_at(t))).collect();
            let interior: Vec<usize> = mine.iter().copied().filter(|&i| !border.contains(&i)).collect();
            let still = |v: &Vec<usize>| v.iter().filter(|&&i| mv[i] == STILL && strength[i] > 0).count();
            let nofield = interior.iter().filter(|&&i| field[i] < 0.0).count();
            let capped = mine.iter().filter(|&&i| strength[i] >= 240 && mv[i] == STILL).count();
            let bs: i32 = border.iter().map(|&i| strength[i]).sum();
            let is: i32 = interior.iter().map(|&i| strength[i]).sum();
            let walls = (0..n).filter(|&t| is_wall(t) && nb[t].iter().any(|&j| mine_at(j))).count();
            let fights = (0..n).filter(|&t| is_fight(t) && nb[t].iter().any(|&j| mine_at(j))).count();
            let fight_seeds = mine.iter().filter(|&&i| nb[i].iter().any(|&t| is_fight(t))).count();
            let any_ffight = mine.iter().filter(|&&i| ffight[i]).count();
            let maxf = mine.iter().map(|&i| field[i]).fold(f64::MIN, f64::max);
            let fvals: Vec<String> = (0..n).filter(|&t| is_fight(t) && nb[t].iter().any(|&j| mine_at(j))).map(|t| format!("{:.2}", tval[t])).collect();
            eprintln!("DBG fight-adjacent pieces {} ffight {} max field {:.2} fight tvals {:?}", fight_seeds, any_ffight, maxf, fvals);
            let to_fight = interior.iter().filter(|&&i| mv[i] != STILL && ffight[i]).count();
            let to_land = interior.iter().filter(|&&i| mv[i] != STILL && !ffight[i]).count();
            let fight_str: i32 = interior.iter().filter(|&&i| ffight[i]).map(|&i| strength[i]).sum();
            eprintln!("border {} (str {}, still {}) interior {} (str {}, still {}, no field {}) capped-still {} walls {} fights {} | moving to fight {} to land {} | interior str pulled by fights {}",
                border.len(), bs, still(&border), interior.len(), is, still(&interior), nofield, capped, walls, fights, to_fight, to_land, fight_str);
        }
        mine.iter()
            .map(|&i| Move { loc: Location { x: (i % self.w) as u16, y: (i / self.w) as u16 }, direction: mv[i] })
            .collect()
    }
}

/// Pieces already moving keep their move; packs only recruit idle pieces.
fn done_fight(mv: &[u8], j: usize) -> bool {
    mv[j] != STILL
}

fn main() {
    let (my_id, game_map, mut conn) = get_init();
    send_init("ApexAlpha");

    let (w, h) = (game_map.width as usize, game_map.height as usize);
    let n = w * h;
    let prod: Vec<i32> = (0..n).map(|i| game_map.contents[i / w][i % w].production as i32).collect();
    let nb: Vec<[usize; 4]> = (0..n)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            [((y + h - 1) % h) * w + x, y * w + (x + 1) % w, ((y + 1) % h) * w + x, y * w + (x + w - 1) % w]
        })
        .collect();
    let mut region = vec![0.0f64; n];
    const R: i64 = 4;
    for i in 0..n {
        let (x, y) = ((i % w) as i64, (i / w) as i64);
        let (mut sum, mut wsum) = (0.0, 0.0);
        for dy in -R..=R {
            for dx in -R..=R {
                let d = dx.abs() + dy.abs();
                if d == 0 || d > R {
                    continue;
                }
                let j = (((y + dy).rem_euclid(h as i64)) as usize) * w + ((x + dx).rem_euclid(w as i64)) as usize;
                let wt = 1.0 / (1.0 + d as f64);
                sum += wt * prod[j] as f64;
                wsum += wt;
            }
        }
        region[i] = sum / wsum;
    }
    let bot = Bot { me: my_id as i32, w, n, prod, region, nb, p: Params::load() };

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
