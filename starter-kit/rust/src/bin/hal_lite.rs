//! HAL-lite: a Rust port of David Ma's (djma) Halite I bot.
//!
//! Source: https://github.com/djma/halite-bot, `HAL-lite/MyBot.java` (v15b). He wrote it for Two
//! Sigma's internal summer contest in 2016 and submitted it unchanged to the open Halite I
//! competition, where it finished 19th. Port of the final version, move for move, including its
//! quirks (they are part of how it plays):
//!
//! - Two modes. Expansion mode (until any piece first sees an enemy in reach, and only while the
//!   frontier is expensive: production / strength < 0.2) runs a greedy planner: every frontier
//!   site gets a two-step production / strength score, a BFS inward (up to 8 steps) finds the
//!   pieces that can take it, the best candidate is committed, its pieces are marked used, and
//!   the loop repeats until no candidate is left or 900 ms are spent. Other interior pieces walk
//!   straight to the border.
//! - Normal mode: border pieces take the best production / strength neighbour or attack the
//!   zero-strength gap that hits the most enemy strength (a "triangle" of the 8 sites in reach),
//!   interior pieces follow a 4-direction diffusion map toward the border, 255 strength is
//!   enlisted toward every fight, a repulsion map keeps big pieces apart, collisions over the cap
//!   are resolved, and still pieces step out of the way of an incoming 255.
//!
//! The Java bot iterates `HashMap<Location, ...>` in several places where order matters, and
//! `Location.hashCode()` is `x * 1000 + y`. `JMap` reproduces Java's HashMap iteration order
//! (bucket `hash & (capacity - 1)`, insertion order within a bucket, capacity doubling past 0.75
//! load and kept by `clear()`), so the port visits sites in the same order as the original.
//! Differences: the original shuffled directions with an unseeded `java.util.Random`; this port
//! uses a generator seeded from the map, so games are reproducible. The 900 ms expansion budget
//! is 800 ms here (`HAL_TIME_MS`), measured from when the frame was read.

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::time::Instant;

use hlt::*;

const MOVE_EFFICIENCY_MINIMUM: f64 = 0.8;
const REV: [u8; 5] = [STILL, SOUTH, WEST, NORTH, EAST];
const LEFT: [u8; 5] = [STILL, WEST, NORTH, EAST, SOUTH];
const RIGHT: [u8; 5] = [STILL, EAST, SOUTH, WEST, NORTH];

/// A `java.util.HashMap<Location, V>` keyed by site index, iterated in Java's order.
struct JMap<V: Copy> {
    hash: Vec<u32>,
    cap: usize,
    seq: u64,
    vals: Vec<Option<V>>,
    order: Vec<u64>,
    keys: Vec<usize>,
}

impl<V: Copy> JMap<V> {
    fn new(hash: &[u32]) -> JMap<V> {
        JMap {
            hash: hash.to_vec(),
            cap: 16,
            seq: 0,
            vals: vec![None; hash.len()],
            order: vec![0; hash.len()],
            keys: Vec::new(),
        }
    }

    fn clear(&mut self) {
        for &k in &self.keys {
            self.vals[k] = None;
        }
        self.keys.clear();
    }

    fn put(&mut self, k: usize, v: V) {
        if self.vals[k].is_none() {
            self.keys.push(k);
            self.order[k] = self.seq;
            self.seq += 1;
            if self.keys.len() > self.cap * 3 / 4 {
                self.cap *= 2;
            }
        }
        self.vals[k] = Some(v);
    }

    fn has(&self, k: usize) -> bool {
        self.vals[k].is_some()
    }

    fn get(&self, k: usize) -> V {
        self.vals[k].unwrap()
    }

    /// Keys in Java's iteration order.
    fn keys(&self) -> Vec<usize> {
        let mut ks = self.keys.clone();
        let mask = self.cap - 1;
        ks.sort_by_key(|&k| ((self.hash[k] as usize) & mask, self.order[k]));
        ks
    }
}

/// `Double.compare` (NaN above everything, -0.0 below 0.0).
fn java_double_cmp(a: f64, b: f64) -> Ordering {
    if a < b {
        return Ordering::Less;
    }
    if a > b {
        return Ordering::Greater;
    }
    let bits = |v: f64| if v.is_nan() { 0x7ff8_0000_0000_0000u64 as i64 } else { v.to_bits() as i64 };
    bits(a).cmp(&bits(b))
}

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

#[derive(Clone, Copy)]
struct Outer {
    dir: u8,
    fighting: bool,
}

#[derive(Clone, Copy)]
struct Inner {
    dir: u8,
    dist: i32,
}

struct ExpansionData {
    visited: Vec<usize>,
    two_step_strength: i32,
    two_step_prod: i32,
    prod_loss: i32,
    mv: (usize, u8),
}

struct LocNode {
    loc: usize,
    visited: Vec<usize>,
    target: i32,
    path_prod: i32,
    depth: i32,
    last_dir: u8,
    prod_loss: i32,
}

struct Bot {
    w: usize,
    h: usize,
    me: u8,
    owner: Vec<u8>,
    strength: Vec<i32>,
    prod: Vec<i32>,
    nbr: Vec<[usize; 5]>,
    outer: JMap<Outer>,
    nml: JMap<()>,
    inner: JMap<Inner>,
    repulsion: Vec<u8>,
    enlisted: JMap<Inner>,
    candidates_hash: Vec<u32>,
    nml_strg: i32,
    nml_prod: i32,
    expansion_mode: bool,
    rng: Rng,
    start: Instant,
    budget_ms: u128,
}

impl Bot {
    fn loc(&self, i: usize, d: u8) -> usize {
        self.nbr[i][d as usize]
    }

    fn shuffled_cardinals(&mut self) -> [u8; 4] {
        // Collections.shuffle: for i from size-1 down to 1, swap(i, rnd.nextInt(i + 1))
        let mut c = CARDINALS;
        for i in (1..4).rev() {
            let j = self.rng.below(i as u64 + 1) as usize;
            c.swap(i, j);
        }
        c
    }

    fn initial_sweep(&mut self) {
        for i in 0..self.w * self.h {
            if self.owner[i] == self.me {
                for &d in &CARDINALS {
                    let n = self.loc(i, d);
                    if self.owner[n] != self.me {
                        self.outer.put(i, Outer { dir: STILL, fighting: false });
                        self.nml.put(n, ());
                        self.nml_strg += self.strength[n];
                        self.nml_prod += self.prod[n];
                    }
                }
                if !self.outer.has(i) {
                    self.inner.put(i, Inner { dir: STILL, dist: i32::MAX });
                }
            }
        }
    }

    fn two_step_best(&self, l: usize) -> (i32, i32) {
        let mut best = 0.0f64;
        let (mut cp, mut cs) = (0, 0);
        for &d in &CARDINALS {
            let ss = self.loc(l, d);
            if self.owner[ss] == 0 {
                for &dd in &CARDINALS {
                    let sss = self.loc(ss, dd);
                    if dd != d && self.owner[sss] == 0 {
                        let p = self.prod[l] + self.prod[ss] + self.prod[sss];
                        let s = self.strength[l] + self.strength[ss] + self.strength[sss] + self.prod[l] + self.prod[ss] + 1;
                        let r = p as f64 / s as f64;
                        if r > best {
                            best = r;
                            cp = p;
                            cs = s;
                        }
                    }
                }
            }
        }
        (cp, cs)
    }

    fn compute_expansion_data(&self, cand: usize, used: &[bool]) -> ExpansionData {
        const MAX_DEPTH: i32 = 8;
        const TOLERABLE_WAIT: i32 = 3;
        let (two_step_prod, two_step_strength) = self.two_step_best(cand);
        let mut final_visited = Vec::new();
        let mut prod_loss = i32::MAX;
        let mut mv = (cand, STILL);
        let target_strg = self.strength[cand];
        let mut queue: VecDeque<LocNode> = VecDeque::new();
        for &d in &CARDINALS {
            let n = self.loc(cand, d);
            if self.inner.has(n) && !used[n] {
                queue.push_back(LocNode {
                    loc: n,
                    visited: vec![n],
                    target: target_strg,
                    path_prod: 0,
                    depth: 1,
                    last_dir: d,
                    prod_loss: self.prod[n],
                });
            }
        }
        while let Some(mut cur) = queue.pop_front() {
            let cs = self.strength[cur.loc];
            if cur.target < cs {
                final_visited = cur.visited;
                mv = (cur.loc, REV[cur.last_dir as usize]);
                prod_loss = cur.prod_loss;
                break;
            }
            for &d in &CARDINALS {
                let n = self.loc(cur.loc, d);
                if self.inner.has(n) && !used[n] && !cur.visited.contains(&n) && cur.depth < MAX_DEPTH {
                    cur.visited.push(n);
                    queue.push_back(LocNode {
                        loc: n,
                        visited: cur.visited.clone(),
                        // Java uses the candidate's strength here, not cur.target
                        target: target_strg - (cur.path_prod + cs),
                        path_prod: cur.path_prod + self.prod[cur.loc],
                        depth: cur.depth + 1,
                        last_dir: d,
                        prod_loss: cur.prod_loss + self.prod[n],
                    });
                }
            }
            if queue.is_empty() && cur.target < cs + cur.prod_loss * TOLERABLE_WAIT {
                final_visited = cur.visited;
                mv = (cur.loc, STILL);
                prod_loss = cur.prod_loss + (self.prod[cand] - 1) * TOLERABLE_WAIT;
                break;
            }
        }
        ExpansionData { visited: final_visited, two_step_strength, two_step_prod, prod_loss, mv }
    }

    fn greedy_expand_with_prod_loss(&mut self) {
        for l in self.outer.keys() {
            self.inner.put(l, Inner { dir: STILL, dist: i32::MAX });
        }
        let n = self.w * self.h;
        let mut selected = vec![false; n];
        let mut selected_moves: Vec<(usize, u8)> = Vec::new();
        let mut used = vec![false; n];
        let mut candidates: JMap<()> = JMap::new(&self.candidates_hash);
        let mut cand_data: Vec<Option<ExpansionData>> = (0..n).map(|_| None).collect();
        let nml_keys = self.nml.keys();
        loop {
            candidates.clear();
            let mut has_valid = false;
            for &c in &nml_keys {
                if !selected[c] {
                    let ed = self.compute_expansion_data(c, &used);
                    if !ed.visited.is_empty() {
                        has_valid = true;
                        candidates.put(c, ());
                        cand_data[c] = Some(ed);
                    }
                }
            }
            if has_valid {
                // Stable descending sort by value; take the first.
                let value = |ed: &ExpansionData| ed.two_step_prod as f64 / (ed.two_step_strength + ed.prod_loss.wrapping_mul(3)) as f64;
                let mut best: Option<(usize, f64)> = None;
                for c in candidates.keys() {
                    let v = value(cand_data[c].as_ref().unwrap());
                    match best {
                        None => best = Some((c, v)),
                        Some((_, bv)) => {
                            if java_double_cmp(v, bv) == Ordering::Greater {
                                best = Some((c, v));
                            }
                        }
                    }
                }
                let (b, _) = best.unwrap();
                selected[b] = true;
                let ed = cand_data[b].as_ref().unwrap();
                selected_moves.push(ed.mv);
                for &l in &ed.visited {
                    used[l] = true;
                }
            }
            if !(has_valid && self.start.elapsed().as_millis() < self.budget_ms) {
                break;
            }
        }
        for (l, d) in selected_moves {
            self.inner.put(l, Inner { dir: d, dist: 0 });
        }
    }

    fn resolve_collisions_helper(&mut self, to_resolve: &mut VecDeque<usize>) {
        while let Some(l) = to_resolve.pop_front() {
            let mut cur = 0;
            if (self.outer.has(l) && self.outer.get(l).dir == STILL) || (self.inner.has(l) && self.inner.get(l).dir == STILL) {
                cur = self.strength[l] + self.prod[l];
            }
            for &d in &CARDINALS {
                let c = self.loc(l, d);
                let cs = self.strength[c];
                if self.inner.has(c) && self.inner.get(c).dir == REV[d as usize] {
                    if (255.min(cur + cs) - cur) as f64 / (cs + 1) as f64 > MOVE_EFFICIENCY_MINIMUM {
                        cur += cs;
                    } else {
                        let dist = self.inner.get(c).dist;
                        self.inner.put(c, Inner { dir: STILL, dist });
                        to_resolve.push_back(c);
                    }
                }
                if self.outer.has(c) && self.outer.get(c).dir == REV[d as usize] {
                    if (255.min(cur + cs) - cur) as f64 / (cs + 1) as f64 > MOVE_EFFICIENCY_MINIMUM {
                        cur += cs;
                    } else {
                        let fighting = self.outer.get(c).fighting;
                        self.outer.put(c, Outer { dir: STILL, fighting });
                        to_resolve.push_back(c);
                    }
                }
            }
        }
    }

    fn resolve_collisions(&mut self) {
        for l in self.nml.keys() {
            let mut cur = 0;
            for &d in &CARDINALS {
                let c = self.loc(l, d);
                let cs = self.strength[c];
                if self.outer.has(c) && self.outer.get(c).dir == REV[d as usize] {
                    if (255.min(cur + cs) - cur) as f64 / (cs + 1) as f64 > MOVE_EFFICIENCY_MINIMUM {
                        cur += cs;
                    } else {
                        let fighting = self.outer.get(c).fighting;
                        self.outer.put(c, Outer { dir: STILL, fighting });
                    }
                }
            }
        }
        let mut to_resolve: VecDeque<usize> = self.outer.keys().into_iter().collect();
        self.resolve_collisions_helper(&mut to_resolve);
        to_resolve.extend(self.inner.keys());
        self.resolve_collisions_helper(&mut to_resolve);
    }

    fn dont_get_gobbled_up(&mut self) {
        for l in self.inner.keys() {
            let m = self.inner.get(l);
            if m.dir == STILL {
                for &d in &CARDINALS {
                    let inc = self.loc(l, d);
                    if self.inner.has(inc)
                        && self.inner.get(inc).dir == REV[d as usize]
                        && self.strength[inc] >= 255 - 5 * self.prod[l]
                        && self.strength[l] > 0
                    {
                        self.inner.put(l, Inner { dir: REV[d as usize], dist: m.dist });
                        break;
                    }
                }
            }
        }
    }

    fn enemy_at(&self, i: usize) -> bool {
        self.owner[i] != self.me && self.owner[i] != 0
    }

    fn outer_ring_strategy(&mut self) {
        for l in self.outer.keys() {
            let mut highest_ratio = 0.0f64;
            let mut ratio_dir = STILL;
            let mut enemy_dir = STILL;
            let mut highest_damage = 0;
            let cur_s = self.strength[l];
            let mut next_to_inner_255 = false;
            let mut fighting = false;
            for &d in &CARDINALS {
                let t = self.loc(l, d);
                if self.inner.has(t) && self.strength[t] == 255 {
                    next_to_inner_255 = true;
                    break;
                }
            }
            for &d in &CARDINALS {
                let t = self.loc(l, d);
                let move_condition = cur_s > self.strength[t] || next_to_inner_255;
                if self.owner[t] != self.me && move_condition {
                    let r = self.prod[t] as f64 / (self.strength[t] + 1) as f64 + 0.0001;
                    if r > highest_ratio {
                        ratio_dir = d;
                        highest_ratio = self.prod[t] as f64 / (self.strength[t] + 1) as f64;
                    }
                }
                let mut damage = 0;
                if self.owner[t] != self.me && self.strength[t] == 0 {
                    fighting = true;
                    let (dl, dr) = (LEFT[d as usize], RIGHT[d as usize]);
                    let td = self.loc(t, d);
                    let tl = self.loc(t, dl);
                    let tr = self.loc(t, dr);
                    // (site, adds production, weight)
                    let ring = [
                        (td, true, 2),
                        (tl, true, 2),
                        (tr, true, 2),
                        (self.loc(td, d), false, 1),
                        (self.loc(td, dl), false, 2),
                        (self.loc(td, dr), false, 2),
                        (self.loc(tl, dl), false, 1),
                        (self.loc(tr, dr), false, 1),
                    ];
                    for &(e, with_prod, wgt) in &ring {
                        if self.enemy_at(e) {
                            let es = self.strength[e] + if with_prod { self.prod[e] } else { 0 };
                            damage += cur_s.min(es) * wgt;
                        }
                    }
                    if damage > highest_damage {
                        enemy_dir = d;
                        highest_damage = damage;
                    }
                }
            }
            if enemy_dir != STILL {
                self.outer.put(l, Outer { dir: enemy_dir, fighting });
                self.expansion_mode = false;
            } else {
                self.outer.put(l, Outer { dir: ratio_dir, fighting });
            }
        }
    }

    fn generate_inner_area_diffusion_map(&mut self, aggressivity: i32) {
        let dirs = self.shuffled_cardinals();
        let nml = self.nml.keys();
        for &d in &dirs {
            for &l in &nml {
                let (s, p) = (self.strength[l], self.prod[l]);
                let mut new_dist = 1 + match aggressivity {
                    3 => s / (p + 1),
                    2 => s / (2 * p + 1),
                    1 => s / (3 * p + 1),
                    _ => 0,
                };
                let mut nl = self.loc(l, d);
                while self.inner.has(nl) || self.outer.has(nl) {
                    if self.inner.has(nl) && new_dist < self.inner.get(nl).dist && self.strength[nl] >= 6 * self.prod[nl] {
                        self.inner.put(nl, Inner { dir: REV[d as usize], dist: new_dist });
                    }
                    new_dist += 1;
                    nl = self.loc(nl, d);
                }
            }
        }
    }

    fn move_unused_tiles(&mut self) {
        let dirs = self.shuffled_cardinals();
        let nml = self.nml.keys();
        for &d in &dirs {
            for &l in &nml {
                let mut nl = self.loc(l, d);
                let mut new_dist = 0;
                while self.inner.has(nl) || self.outer.has(nl) {
                    // usedForExpansion is never filled in the original, so it is not checked here
                    if self.inner.has(nl)
                        && new_dist < self.inner.get(nl).dist
                        && self.strength[nl] >= 5 * self.prod[nl]
                        && !self.outer.has(nl)
                    {
                        self.inner.put(nl, Inner { dir: REV[d as usize], dist: new_dist });
                    }
                    new_dist += 1;
                    nl = self.loc(nl, d);
                }
            }
        }
    }

    fn enlistable(&self, i: usize) -> bool {
        self.inner.has(i) || (self.outer.has(i) && !self.outer.get(i).fighting)
    }

    fn enlist_combat(&mut self) {
        let n = self.w * self.h;
        let mut visited = vec![false; n];
        for l in self.outer.keys() {
            if !self.outer.get(l).fighting {
                continue;
            }
            let mut to_enlist_strength = 255;
            let mut queue: VecDeque<(usize, u8)> = VecDeque::new();
            let mut touched: Vec<usize> = Vec::new();
            for &d in &CARDINALS {
                let nx = self.loc(l, d);
                if !visited[nx] && self.enlistable(nx) {
                    queue.push_back((nx, REV[d as usize]));
                }
            }
            while let Some((vl, vd)) = queue.pop_front() {
                to_enlist_strength -= self.strength[vl];
                let tgt = self.loc(vl, vd);
                let depth = if self.enlisted.has(tgt) { self.enlisted.get(tgt).dist } else { 0 };
                if self.strength[vl] > 5 * self.prod[vl] && (!self.enlisted.has(vl) || depth + 1 < self.enlisted.get(vl).dist) {
                    self.enlisted.put(vl, Inner { dir: vd, dist: depth + 1 });
                }
                if to_enlist_strength <= 0 {
                    break;
                }
                visited[vl] = true;
                touched.push(vl);
                for &d in &CARDINALS {
                    let nx = self.loc(vl, d);
                    if !visited[nx] && self.enlistable(nx) {
                        queue.push_back((nx, REV[d as usize]));
                    }
                }
            }
            for t in touched {
                visited[t] = false;
            }
        }
        for l in self.enlisted.keys() {
            let e = self.enlisted.get(l);
            if self.outer.has(l) {
                self.outer.put(l, Outer { dir: e.dir, fighting: false });
            } else if self.inner.has(l) {
                let dist = self.inner.get(l).dist;
                self.inner.put(l, Inner { dir: e.dir, dist });
            }
        }
    }

    fn generate_repulsion_map(&mut self) {
        for i in 0..self.w * self.h {
            self.repulsion[i] = STILL;
            if !(self.inner.has(i) || self.outer.has(i)) {
                continue;
            }
            let s = self.strength[i];
            let mut ry = 0;
            let mut rx = 0;
            ry -= s.min(self.strength[self.loc(i, NORTH)]);
            ry += s.min(self.strength[self.loc(i, SOUTH)]);
            rx -= s.min(self.strength[self.loc(i, EAST)]);
            rx += s.min(self.strength[self.loc(i, WEST)]);
            let p5 = 5 * self.prod[i];
            if rx.abs() >= ry.abs().max(p5) {
                self.repulsion[i] = if rx > 0 { EAST } else { WEST };
            } else if ry.abs() > rx.abs().max(p5) {
                self.repulsion[i] = if ry > 0 { NORTH } else { SOUTH };
            }
        }
    }

    fn resolve_ensemble_map(&mut self) {
        for l in self.inner.keys() {
            let m = self.inner.get(l);
            let r = self.repulsion[l];
            if r != STILL && m.dir == REV[r as usize] && self.outer.has(self.loc(l, m.dir)) {
                self.inner.put(l, Inner { dir: STILL, dist: m.dist });
            }
        }
        for l in self.outer.keys() {
            let m = self.outer.get(l);
            let r = self.repulsion[l];
            if r != STILL && m.dir == REV[r as usize] {
                let t = self.loc(l, m.dir);
                if self.outer.has(t) && self.outer.get(t).fighting {
                    self.outer.put(l, Outer { dir: STILL, fighting: m.fighting });
                }
            }
        }
    }

    fn turn(&mut self, map: &GameMap) -> Vec<Move> {
        self.start = Instant::now();
        for y in 0..self.h {
            for x in 0..self.w {
                let s = map.contents[y][x];
                let i = y * self.w + x;
                self.owner[i] = s.owner;
                self.strength[i] = s.strength as i32;
            }
        }
        self.outer.clear();
        self.nml.clear();
        self.inner.clear();
        self.enlisted.clear();
        self.nml_strg = 0;
        self.nml_prod = 0;

        self.initial_sweep();
        self.outer_ring_strategy();

        let mut out: Vec<(usize, u8)> = Vec::new();
        if self.expansion_mode && (self.nml_prod as f64) / ((self.nml_strg + 1) as f64) < 0.2 {
            self.greedy_expand_with_prod_loss();
            self.move_unused_tiles();
            for l in self.inner.keys() {
                out.push((l, self.inner.get(l).dir));
            }
        } else {
            self.generate_inner_area_diffusion_map(1);
            self.enlist_combat();
            self.generate_repulsion_map();
            self.resolve_ensemble_map();
            self.resolve_collisions();
            self.dont_get_gobbled_up();
            for l in self.outer.keys() {
                out.push((l, self.outer.get(l).dir));
            }
            for l in self.inner.keys() {
                out.push((l, self.inner.get(l).dir));
            }
        }
        out.into_iter()
            .map(|(i, d)| Move { loc: Location { x: (i % self.w) as u16, y: (i / self.w) as u16 }, direction: d })
            .collect()
    }
}

fn main() {
    let (me, init_map, mut conn) = get_init();
    send_init("HAL-lite");
    let (w, h) = (init_map.width as usize, init_map.height as usize);
    let n = w * h;
    let mut prod = vec![0; n];
    let mut nbr = vec![[0usize; 5]; n];
    let mut hash = vec![0u32; n];
    let mut seed = 0x9e37_79b9_7f4a_7c15u64 ^ me as u64;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let s = init_map.contents[y][x];
            prod[i] = s.production as i32;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(s.production as u64 * 256 + s.strength as u64 + 1);
            let l = Location { x: x as u16, y: y as u16 };
            for &d in &DIRECTIONS {
                let m = init_map.get_location(l, d);
                nbr[i][d as usize] = m.y as usize * w + m.x as usize;
            }
            let hc = (x as i32).wrapping_mul(1000).wrapping_add(y as i32) as u32;
            hash[i] = hc ^ (hc >> 16);
        }
    }
    let budget_ms = std::env::var("HAL_TIME_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(800);
    let mut bot = Bot {
        w,
        h,
        me,
        owner: vec![0; n],
        strength: vec![0; n],
        prod,
        nbr,
        outer: JMap::new(&hash),
        nml: JMap::new(&hash),
        inner: JMap::new(&hash),
        repulsion: vec![STILL; n],
        enlisted: JMap::new(&hash),
        candidates_hash: hash.clone(),
        nml_strg: 0,
        nml_prod: 0,
        expansion_mode: true,
        rng: Rng(seed | 1),
        start: Instant::now(),
        budget_ms,
    };
    loop {
        let map = conn.get_frame();
        let moves = bot.turn(&map);
        conn.send_frame(&moves);
    }
}
