//! ApexGamma: ApexBeta with a joint search of front pieces and a stronger
//! pull toward fights.
//!
//! ApexBeta (see apex_beta.rs) runs the ApexAlpha policy, then refines every
//! piece within 3 steps of an enemy with an exact one-turn simulation, enemies
//! predicted by running the same policy from their seat. Its search changed one
//! piece at a time, so it could not find moves that only pay when two pieces
//! act together (two pieces stepping into a gap at once). ApexGamma adds:
//!
//!   1. Pair search. After each single-piece pass, every pair of front pieces
//!      within 2 steps of each other tries all joint moves. 59% of 1v1s
//!      against ApexBeta (1000 games; 48% with pairs off on the same seeds).
//!   2. Fight value 3 (was 1.5): interior strength answers fights harder.
//!      About +3 points on top of pairs.
//!
//! Tried and switched off (still available): triples (APEX_TRIPLES=1, no
//! gain over pairs), per-opponent learned follow rates with sampled
//! scenarios (APEX_SAMPLES=8: 44% vs ApexBeta, worse than Beta's two
//! correlated scenarios). Lower weight on enemy losses in FFA (APEX_LAMBDA
//! 0.5-0.7) is much worse in every format.
//!
//! Every constant reads an APEX_* environment variable (see Params::load).

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
    combat: f64,      // one-turn combat search at the front (0 = off: plays as ApexAlpha)
    zone: f64,        // pieces this close to an enemy take part in the search
    pred: f64,        // weight of "enemies play our policy" (the rest: "enemies stay")
    lambda: f64,      // enemy strength destroyed, against ours lost
    lambda_terr: f64, // enemy sites held count this much against ours
    terr: f64,        // a site held is worth terr * (production + 0.5) strength
    neutral_w: f64,   // strength spent on neutral sites counts this much as a loss
    bias: f64,        // margin needed to leave the default (ApexAlpha) move
    passes: f64,      // coordinate ascent passes
    depth: f64,       // 1: enemies predicted with the policy; 2: with policy + their own search
    samples: f64,     // sampled scenarios, each enemy piece following its prediction independently (0: Beta's two)
    follow: f64,      // prior chance that a piece predicted to move does move...
    prior: f64,       // ...worth this many observations; then learnt per opponent
    pairs: f64,       // also search joint moves of front pieces within 2 steps (0 = off)
    triples: f64,     // ...and of three pieces all within 2 steps of each other (0 = off)
    max_triples: f64, // at most this many triples per turn (time limit)
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
            fight_base: env("APEX_FIGHT", 3.0),
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
            combat: env("APEX_COMBAT", 1.0),
            zone: env("APEX_ZONE", 3.0),
            pred: env("APEX_PRED", 0.6),
            lambda: env("APEX_LAMBDA", 1.0),
            lambda_terr: env("APEX_LAMBDA_TERR", 1.0),
            terr: env("APEX_TERR", 2.0),
            neutral_w: env("APEX_NEUTRAL_W", 0.3),
            bias: env("APEX_BIAS", 1.0),
            passes: env("APEX_PASSES", 3.0),
            depth: env("APEX_DEPTH", 1.0),
            samples: env("APEX_SAMPLES", 0.0),
            follow: env("APEX_FOLLOW", 0.6),
            prior: env("APEX_PRIOR", 20.0),
            pairs: env("APEX_PAIRS", 1.0),
            triples: env("APEX_TRIPLES", 0.0),
            max_triples: env("APEX_MAX_TRIPLES", 400.0),
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
    /// What we predicted each enemy piece would do last turn, and the state
    /// then, to score the prediction once the next frame shows what happened.
    last_pred: Vec<u8>,
    last_owner: Vec<i32>,
    last_strength: Vec<i32>,
    /// Per player, enemy pieces near us: [predicted to move, of which moved,
    /// predicted to stay, of which moved].
    seen: Vec<[f64; 4]>,
    turn_no: u64,
}

impl Bot {
    /// ApexAlpha's whole policy, seen from player `me`: one direction per site.
    /// Also used to predict what each enemy will do.
    fn policy(&self, me: i32, owner: &[i32], strength: &[i32]) -> Vec<u8> {
        let (n, prod, nb, p) = (self.n, &self.prod, &self.nb, &self.p);
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

        mv
    }
}

/// One-turn combat simulation around the front (see the module comment).
struct Sim<'a> {
    bot: &'a Bot,
    me: i32,
    owner: &'a [i32],
    strength: &'a [i32],
    /// Our strength arriving on (or staying at) each site, before the cap.
    o_raw: Vec<i32>,
    /// Our pieces moving onto each site (a site we own always holds a piece).
    arr: Vec<i32>,
    /// Per scenario: enemy strength on each site after their moves (-1: none),
    /// and the enemy strength that hits a piece standing there.
    e_s: Vec<Vec<i32>>,
    e_hit: Vec<Vec<i32>>,
    weight: Vec<f64>,
}

impl<'a> Sim<'a> {
    fn dest(&self, i: usize, d: u8) -> usize {
        if d == STILL { i } else { self.bot.nb[i][d as usize - 1] }
    }
    fn ours(&self, c: usize) -> i32 {
        if self.owner[c] == self.me || self.arr[c] > 0 { self.o_raw[c].min(255) } else { -1 }
    }
    fn apply(&mut self, i: usize, d: u8, sign: i32) {
        let s = self.strength[i] + if d == STILL { self.bot.prod[i] } else { 0 };
        let t = self.dest(i, d);
        self.o_raw[t] += sign * s;
        if d != STILL {
            self.arr[t] += sign;
        }
    }
    fn cell(&self, c: usize, k: usize) -> f64 {
        let p = &self.bot.p;
        let os = self.ours(c);
        let es = self.e_s[k][c];
        if os < 0 && es < 0 {
            return 0.0;
        }
        let ns = if self.owner[c] == 0 { self.strength[c] } else { 0 };
        let mut score = 0.0;
        let mut us_alive = false;
        if os >= 0 {
            let hit = self.e_hit[k][c];
            let du = hit + ns;
            score -= (os.min(hit)) as f64 + p.neutral_w * (os - hit).max(0).min(ns) as f64;
            us_alive = os - du > 0 || (os == 0 && du == 0);
            if self.o_raw[c] > 255 {
                score -= (self.o_raw[c] - 255) as f64;
            }
        }
        let mut en_alive = false;
        if es >= 0 {
            let mut de = ns;
            de += self.ours(c).max(0);
            for &j in &self.bot.nb[c] {
                de += self.ours(j).max(0);
            }
            score += p.lambda * es.min(de) as f64;
            en_alive = es - de > 0 || (es == 0 && de == 0);
        }
        let v = p.terr * (self.bot.prod[c] as f64 + 0.5);
        if us_alive && !en_alive {
            score += v;
        }
        if en_alive && !us_alive {
            score -= p.lambda_terr * v;
        }
        score
    }
    /// Weighted score of the sites whose outcome depends on these sites' contents.
    fn local(&self, sites: &[usize], buf: &mut Vec<usize>) -> f64 {
        buf.clear();
        for &x in sites {
            if !buf.contains(&x) {
                buf.push(x);
            }
            for &j in &self.bot.nb[x] {
                if !buf.contains(&j) {
                    buf.push(j);
                }
            }
        }
        let mut s = 0.0;
        for k in 0..self.weight.len() {
            if self.weight[k] == 0.0 {
                continue;
            }
            let mut t = 0.0;
            for &c in buf.iter() {
                t += self.cell(c, k);
            }
            s += self.weight[k] * t;
        }
        s
    }
}

impl Bot {
    /// Scores last turn's prediction: a piece that moved leaves a strength-0
    /// site of its owner behind; a piece that died tells us nothing.
    fn observe(&mut self, owner: &[i32], strength: &[i32]) {
        for i in 0..self.n {
            let q = self.last_owner[i];
            if q == 0 || q == self.me || self.last_strength[i] == 0 || self.last_pred[i] == u8::MAX || owner[i] != q {
                continue;
            }
            let moved = strength[i] == 0;
            if self.seen.len() <= q as usize {
                self.seen.resize(q as usize + 1, [0.0; 4]);
            }
            let st = &mut self.seen[q as usize];
            let k = if self.last_pred[i] != STILL { 0 } else { 2 };
            st[k] += 1.0;
            if moved {
                st[k + 1] += 1.0;
            }
        }
    }

    /// Learned chance that a piece of player q predicted to move really moves
    /// (prior: APEX_FOLLOW over APEX_PRIOR pseudo-observations).
    fn follow(&self, q: i32) -> f64 {
        let (a, n) = self.seen.get(q as usize).map_or((0.0, 0.0), |st| (st[1], st[0]));
        (a + self.p.follow * self.p.prior) / (n + self.p.prior)
    }

    fn turn(&mut self, owner: &[i32], strength: &[i32]) -> Vec<Move> {
        self.turn_no += 1;
        self.observe(owner, strength);
        let mut pred = vec![u8::MAX; self.n];
        let mv = self.decide(self.me, owner, strength, self.p.depth as u32, Some(&mut pred));
        if std::env::var("APEX_DEBUG").is_ok() {
            for (q, st) in self.seen.iter().enumerate() {
                if st[0] + st[2] > 0.0 {
                    eprintln!("player {} pred-move {} moved {:.3} | pred-stay {} moved {:.3} | follow {:.3}",
                        q, st[0], st[1] / st[0].max(1.0), st[2], st[3] / st[2].max(1.0), self.follow(q as i32));
                }
            }
        }
        self.last_pred = pred;
        self.last_owner = owner.to_vec();
        self.last_strength = strength.to_vec();
        (0..self.n)
            .filter(|&i| owner[i] == self.me)
            .map(|i| Move { loc: Location { x: (i % self.w) as u16, y: (i / self.w) as u16 }, direction: mv[i] })
            .collect()
    }

    fn bfs(&self, seeds: impl Iterator<Item = usize>) -> Vec<i32> {
        let mut dist = vec![i32::MAX; self.n];
        let mut layer: Vec<usize> = seeds.collect();
        for &i in &layer {
            dist[i] = 0;
        }
        let mut d = 0;
        while !layer.is_empty() {
            d += 1;
            let mut next = Vec::new();
            for &i in &layer {
                for &j in &self.nb[i] {
                    if dist[j] == i32::MAX {
                        dist[j] = d;
                        next.push(j);
                    }
                }
            }
            layer = next;
        }
        dist
    }

    /// Player `me`'s moves: the policy, refined by the combat search, whose
    /// enemy prediction looks one level less deep.
    fn decide(&self, me: i32, owner: &[i32], strength: &[i32], depth: u32, out: Option<&mut Vec<u8>>) -> Vec<u8> {
        let mut mv = self.policy(me, owner, strength);
        if depth > 0 && self.p.combat > 0.0 {
            self.combat(me, owner, strength, &mut mv, depth, out);
        }
        mv
    }

    fn combat(&self, me: i32, owner: &[i32], strength: &[i32], mv: &mut Vec<u8>, depth: u32, out: Option<&mut Vec<u8>>) {
        let (n, p) = (self.n, &self.p);
        let zone = p.zone as i32;
        let edist = self.bfs((0..n).filter(|&i| owner[i] != 0 && owner[i] != me));
        let mut pieces: Vec<usize> = (0..n).filter(|&i| owner[i] == me && strength[i] > 0 && edist[i] <= zone).collect();
        if pieces.is_empty() {
            return;
        }
        pieces.sort_by_key(|&i| (-strength[i], i));
        let mdist = self.bfs((0..n).filter(|&i| owner[i] == me));
        let mut rivals: Vec<i32> = Vec::new();
        for i in 0..n {
            if owner[i] != 0 && owner[i] != me && mdist[i] <= zone && !rivals.contains(&owner[i]) {
                rivals.push(owner[i]);
            }
        }
        rivals.sort();
        // Scenario 0: every enemy stays. Scenario 1: each rival plays our policy.
        let mut pred = vec![STILL; n];
        if p.pred > 0.0 {
            for &q in &rivals {
                let m = self.decide(q, owner, strength, depth - 1, None);
                for i in 0..n {
                    if owner[i] == q {
                        pred[i] = m[i];
                    }
                }
            }
        }
        if let Some(out) = out {
            for i in 0..n {
                if owner[i] != 0 && owner[i] != me && mdist[i] <= zone {
                    out[i] = pred[i];
                }
            }
        }
        // Scenarios. With APEX_SAMPLES = 0, Beta's two: everyone stays, and
        // everyone plays the predicted moves. Otherwise each scenario lets
        // every enemy piece follow its predicted move independently, with the
        // chance learnt for its owner this game.
        let samples = p.samples as usize;
        let ns = if samples == 0 { 2 } else { samples };
        let mut weight = if samples == 0 { vec![1.0 - p.pred, p.pred] } else { vec![1.0 / ns as f64; ns] };
        let mut rng = 0x9E3779B97F4A7C15u64 ^ (self.turn_no.wrapping_mul(0x2545F4914F6CDD1D)) ^ (me as u64);
        let follow: Vec<f64> = (0..=rivals.iter().copied().max().unwrap_or(0) as usize).map(|q| self.follow(q as i32)).collect();
        let mut e_s = vec![vec![-1i32; n]; ns];
        let mut e_hit = vec![vec![0i32; n]; ns];
        for k in 0..ns {
            let mut raw = vec![0i32; n];
            let mut here = vec![false; n];
            for i in 0..n {
                if owner[i] == 0 || owner[i] == me {
                    continue;
                }
                let d = if samples == 0 {
                    if k == 0 { STILL } else { pred[i] }
                } else if pred[i] != STILL && (owner[i] as usize) < follow.len() {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    if ((rng >> 11) as f64 / (1u64 << 53) as f64) < follow[owner[i] as usize] { pred[i] } else { STILL }
                } else {
                    STILL
                };
                here[i] = true;
                if d == STILL {
                    raw[i] += strength[i] + self.prod[i];
                } else {
                    let t = self.nb[i][d as usize - 1];
                    raw[t] += strength[i];
                    here[t] = true;
                }
            }
            for i in 0..n {
                if here[i] {
                    e_s[k][i] = raw[i].min(255);
                }
            }
            for i in 0..n {
                let s = e_s[k][i];
                if s > 0 {
                    e_hit[k][i] += s;
                    for &j in &self.nb[i] {
                        e_hit[k][j] += s;
                    }
                }
            }
        }
        if p.pred <= 0.0 && samples == 0 {
            weight = vec![1.0, 0.0];
        }
        let mut sim = Sim {
            bot: self,
            me,
            owner,
            strength,
            o_raw: vec![0; n],
            arr: vec![0; n],
            e_s,
            e_hit,
            weight,
        };
        for i in 0..n {
            if owner[i] == me {
                sim.apply(i, mv[i], 1);
            }
        }
        let default: Vec<u8> = pieces.iter().map(|&i| mv[i]).collect();
        let mut buf = Vec::new();
        let bias_of = |pi: usize, d: u8, cur: u8| -> f64 {
            // Leaving the default move needs a margin: the default also serves
            // expansion and routing, which this score ignores.
            if d == default[pi] { 0.0 } else if cur == default[pi] { p.bias } else { 0.0 }
        };
        // Pairs of front pieces close enough to act on the same sites.
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        if p.pairs > 0.0 {
            let pos: std::collections::HashMap<usize, usize> = pieces.iter().enumerate().map(|(k, &i)| (i, k)).collect();
            for (ka, &a) in pieces.iter().enumerate() {
                let mut near: Vec<usize> = Vec::new();
                for &j in &self.nb[a] {
                    near.push(j);
                    for &k in &self.nb[j] {
                        near.push(k);
                    }
                }
                near.sort();
                near.dedup();
                for c in near {
                    if let Some(&kb) = pos.get(&c) {
                        if kb > ka {
                            pairs.push((ka, kb));
                        }
                    }
                }
            }
        }
        let mut triples: Vec<(usize, usize, usize)> = Vec::new();
        if p.triples > 0.0 {
            let close: std::collections::HashSet<(usize, usize)> = pairs.iter().copied().collect();
            for &(a, b) in &pairs {
                for c in b + 1..pieces.len() {
                    if close.contains(&(a, c)) && close.contains(&(b, c)) {
                        triples.push((a, b, c));
                    }
                }
            }
            triples.truncate(p.max_triples as usize);
        }
        for _ in 0..p.passes as usize {
            let mut changed = false;
            for (pi, &i) in pieces.iter().enumerate() {
                let cur = mv[i];
                let mut best = cur;
                let mut best_gain = 0.0;
                for d in DIRECTIONS {
                    if d == cur {
                        continue;
                    }
                    let sites = [i, sim.dest(i, cur), sim.dest(i, d)];
                    let before = sim.local(&sites, &mut buf);
                    sim.apply(i, cur, -1);
                    sim.apply(i, d, 1);
                    let after = sim.local(&sites, &mut buf);
                    sim.apply(i, d, -1);
                    sim.apply(i, cur, 1);
                    let gain = after - before - bias_of(pi, d, cur);
                    if gain > best_gain + 1e-9 {
                        best_gain = gain;
                        best = d;
                    }
                }
                if best != cur {
                    sim.apply(i, cur, -1);
                    sim.apply(i, best, 1);
                    mv[i] = best;
                    changed = true;
                }
            }
            // Joint moves: two pieces that only gain by moving together.
            for &(ka, kb) in &pairs {
                let (a, b) = (pieces[ka], pieces[kb]);
                let (ca, cb) = (mv[a], mv[b]);
                let mut best = (ca, cb);
                let mut best_gain = 0.0;
                for da in DIRECTIONS {
                    for db in DIRECTIONS {
                        if da == ca || db == cb {
                            continue; // single changes were tried above
                        }
                        let sites = [a, sim.dest(a, ca), sim.dest(a, da), b, sim.dest(b, cb), sim.dest(b, db)];
                        let before = sim.local(&sites, &mut buf);
                        sim.apply(a, ca, -1);
                        sim.apply(a, da, 1);
                        sim.apply(b, cb, -1);
                        sim.apply(b, db, 1);
                        let after = sim.local(&sites, &mut buf);
                        sim.apply(b, db, -1);
                        sim.apply(b, cb, 1);
                        sim.apply(a, da, -1);
                        sim.apply(a, ca, 1);
                        let gain = after - before - bias_of(ka, da, ca) - bias_of(kb, db, cb);
                        if gain > best_gain + 1e-9 {
                            best_gain = gain;
                            best = (da, db);
                        }
                    }
                }
                if best != (ca, cb) {
                    sim.apply(a, ca, -1);
                    sim.apply(a, best.0, 1);
                    sim.apply(b, cb, -1);
                    sim.apply(b, best.1, 1);
                    mv[a] = best.0;
                    mv[b] = best.1;
                    changed = true;
                }
            }
            // Triples: three pieces all within 2 steps of each other.
            for &(ka, kb, kc) in &triples {
                let ks = [ka, kb, kc];
                let ps = [pieces[ka], pieces[kb], pieces[kc]];
                let cur = [mv[ps[0]], mv[ps[1]], mv[ps[2]]];
                let mut best = cur;
                let mut best_gain = 0.0;
                for da in DIRECTIONS {
                    for db in DIRECTIONS {
                        for dc in DIRECTIONS {
                            let new = [da, db, dc];
                            if (0..3).any(|t| new[t] == cur[t]) {
                                continue; // fewer changes were tried above
                            }
                            let mut sites: Vec<usize> = Vec::with_capacity(9);
                            for t in 0..3 {
                                sites.push(ps[t]);
                                sites.push(sim.dest(ps[t], cur[t]));
                                sites.push(sim.dest(ps[t], new[t]));
                            }
                            let before = sim.local(&sites, &mut buf);
                            for t in 0..3 {
                                sim.apply(ps[t], cur[t], -1);
                                sim.apply(ps[t], new[t], 1);
                            }
                            let after = sim.local(&sites, &mut buf);
                            for t in 0..3 {
                                sim.apply(ps[t], new[t], -1);
                                sim.apply(ps[t], cur[t], 1);
                            }
                            let gain = after - before - (0..3).map(|t| bias_of(ks[t], new[t], cur[t])).sum::<f64>();
                            if gain > best_gain + 1e-9 {
                                best_gain = gain;
                                best = new;
                            }
                        }
                    }
                }
                if best != cur {
                    for t in 0..3 {
                        sim.apply(ps[t], cur[t], -1);
                        sim.apply(ps[t], best[t], 1);
                        mv[ps[t]] = best[t];
                    }
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }
}

/// Pieces already moving keep their move; packs only recruit idle pieces.
fn done_fight(mv: &[u8], j: usize) -> bool {
    mv[j] != STILL
}

fn main() {
    let (my_id, game_map, mut conn) = get_init();
    send_init("ApexGamma");

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
    let mut bot = Bot {
        me: my_id as i32, w, n, prod, region, nb, p: Params::load(),
        last_pred: vec![u8::MAX; n], last_owner: vec![0; n], last_strength: vec![0; n], seen: Vec::new(), turn_no: 0,
    };

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
