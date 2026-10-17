//! DexBot: a Rust port of Declan Groves' (DexGroves) Halite I bot, ranked 7th of the 2016-17
//! competition.
//!
//! Source: https://github.com/DexGroves/halite-bot (HEAD: his README says the submitted bot is
//! commit 780144e plus the non-aggression changes of HEAD, which is HEAD in substance), and his
//! write-up "Halite Bot Breakdown" (dexgroves.github.io, 15 Feb 2017, now only on the Web
//! Archive). The original is Python with numpy and scipy; this port keeps its array semantics:
//! arrays indexed `[x, y]` (so `k = x * height + y`), `np.nonzero` and `argmax` in that order,
//! first index wins ties.
//!
//! Each turn the moves pass through four stages, in order:
//! 1. Combatant: pieces next to a fight (an enemy or a 0-strength neutral at our border) hit the
//!    neighbour with the most splash value; pieces within 8 steps walk to the nearest fight.
//! 2. MoveMaker (expansion): every unowned border site i gets a local value from a Gaussian-blurred
//!    `prod^2 / strength` map and a global "tunnel" value `T_i = sum_j G_j / D_ij` over the
//!    neutral sites j for which i is the closest border site, where D is an all-pairs shortest
//!    path computed once at init on edge costs `int(strength_to / (production_from + 0.1))`
//!    (scipy's sparse Dijkstra: a zero cost means no edge). Each idle piece, strongest first,
//!    takes `argmax_j G_j / (dist + t_ij) + k * T_j / (dist + 10)`.
//! 3. Resolver: a chessboard pattern (only `(x + y + turn)` even sites move into the plan; odd
//!    sites dodge), 255 cap control, greedy two-direction pathing.
//! 4. Nonaggressor: a non-aggression pact. Neutral walls touching an enemy are not broken while we
//!    are weaker, except the weakest one once we are twice as strong.
//!
//! Differences from the original: `random.choice` uses a generator seeded from the map (games are
//! reproducible); numpy's unstable `argsort` is replaced by a stable one (ties are rare and their
//! order in numpy is an implementation detail); Python sets and dicts of moves follow site order;
//! when the original would call `np.min([])` (a crash, the bot would be ejected) this port skips
//! that step.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use hlt::*;

const COM_RADIUS: i64 = 8;
const COMBAT_WAIT: i64 = 4;
const WAIT: i64 = 4;
const BULK_MVMT_OFF: f64 = 10.0;
const STRLIM: i64 = 255;
const INF: u32 = u32::MAX;

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

/// Python dict of moves: origin -> (target, direction), in insertion order.
struct Moveset {
    to_move: Vec<bool>,
    order: Vec<usize>,
    entry: Vec<Option<(usize, Option<u8>)>>,
    move_matrix: Vec<bool>,
}

impl Moveset {
    fn new(owned: &[i64]) -> Moveset {
        let n = owned.len();
        Moveset {
            to_move: owned.iter().map(|&o| o != 0).collect(),
            order: Vec::new(),
            entry: vec![None; n],
            move_matrix: vec![false; n],
        }
    }

    fn add_move(&mut self, a: usize, t: usize, dir: Option<u8>) {
        if self.entry[a].is_none() {
            self.order.push(a);
        }
        self.entry[a] = Some((t, dir));
        self.move_matrix[a] = true;
        self.to_move[a] = false;
    }

    fn set_stays(&mut self) {
        for k in 0..self.to_move.len() {
            if self.to_move[k] && self.entry[k].is_none() {
                self.order.push(k);
                self.entry[k] = Some((k, Some(STILL)));
            }
        }
    }
}

/// Indices sorted by value, descending, as `np.argsort(v)[::-1]` with a stable sort.
fn argsort_desc(v: &[i64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by_key(|&i| v[i]);
    idx.reverse();
    idx
}

struct Dex {
    w: usize,
    h: usize,
    n: usize,
    me: u8,
    turn: i64,
    last_turn: i64,
    glob_k: f64,
    nb: Vec<[usize; 4]>,
    xs: Vec<usize>,
    ys: Vec<usize>,
    prod: Vec<i64>,
    prodc: Vec<i64>,
    path: Vec<u32>,
    gauss_mid: Vec<f64>,
    rng: Rng,
    // per turn
    owners: Vec<u8>,
    strn: Vec<i64>,
    owned: Vec<i64>,
    blank: Vec<i64>,
    enemy: Vec<i64>,
    strnc: Vec<i64>,
    wall: Vec<i64>,
    combat_heur: Vec<i64>,
    ubrdr: Vec<i64>,
    obrdr: Vec<i64>,
    melee: Vec<i64>,
    close: Vec<i64>,
    dfc: Vec<i64>,
    gte_nbr: Vec<bool>,
    mbval: Vec<f64>,
    total_strn: f64,
    num_enemies: i64,
    in_combat: bool,
    safe: Vec<i64>,
}

impl Dex {
    fn dist(&self, a: usize, b: usize) -> i64 {
        let dx = (self.xs[a] as i64 - self.xs[b] as i64).rem_euclid(self.w as i64);
        let dy = (self.ys[a] as i64 - self.ys[b] as i64).rem_euclid(self.h as i64);
        let d = dx.min(self.w as i64 - dx) + dy.min(self.h as i64 - dy);
        d.max(1)
    }

    fn at(&self, x: usize, y: usize) -> usize {
        x * self.h + y
    }

    fn plus(&self, v: &[i64], f: fn(i64, i64) -> i64) -> Vec<i64> {
        (0..self.n)
            .map(|k| self.nb[k].iter().fold(v[k], |acc, &m| f(acc, v[m])))
            .collect()
    }

    /// BFS distance to the sources through owned sites only (-1: unreachable).
    fn friendly_to(&self, sources: &[usize]) -> Vec<i64> {
        let mut dist = vec![-1i64; self.n];
        for &s in sources {
            dist[s] = 0;
        }
        let mut q: VecDeque<usize> = sources.iter().copied().collect();
        while let Some(c) = q.pop_front() {
            let cd = dist[c];
            for &m in &self.nb[c] {
                if self.owned[m] == 0 {
                    continue;
                }
                if dist[m] == -1 || dist[m] > cd + 1 {
                    dist[m] = cd + 1;
                    q.push_back(m);
                }
            }
        }
        dist
    }

    fn read_frame(&mut self, map: &GameMap) {
        for y in 0..self.h {
            for x in 0..self.w {
                let k = self.at(x, y);
                let s = map.contents[y][x];
                self.owners[k] = s.owner;
                self.strn[k] = s.strength as i64;
            }
        }
    }

    fn update(&mut self) {
        self.turn += 1;
        let n = self.n;
        for k in 0..n {
            self.owned[k] = (self.owners[k] == self.me) as i64;
            self.blank[k] = (self.owners[k] == 0) as i64;
            self.enemy[k] = 1 - self.owned[k] - self.blank[k];
            self.strnc[k] = self.strn[k].max(1);
            self.wall[k] = self.blank[k] * (self.strn[k] > 0) as i64;
        }
        let sum = |a: i64, b: i64| a + b;
        let max = |a: i64, b: i64| a.max(b);
        let min = |a: i64, b: i64| a.min(b);
        let es: Vec<i64> = (0..n).map(|k| self.strn[k] * self.enemy[k]).collect();
        let splash_dmg = self.plus(&es, sum);
        let ep: Vec<i64> = (0..n).map(|k| self.prod[k] * self.enemy[k] * (self.strn[k] == 0) as i64).collect();
        let splash_prod = self.plus(&ep, sum);
        self.combat_heur = (0..n)
            .map(|k| {
                let z = (self.strn[k] == 0) as i64;
                splash_dmg[k]
                    + self.prodc[k] * self.blank[k] * z
                    + self.prodc[k] * self.enemy[k] * 2
                    + self.enemy[k] * 2
                    + self.blank[k] * z
                    + splash_prod[k]
            })
            .collect();

        let pm = self.plus(&self.owned, max);
        self.ubrdr = (0..n).map(|k| pm[k] - self.owned[k]).collect();
        let pu = self.plus(&self.ubrdr, max);
        self.obrdr = (0..n).map(|k| pu[k] - self.ubrdr[k]).collect();

        let pe = self.plus(&self.enemy, max);
        let ebrdr: Vec<i64> = (0..n).map(|k| pe[k] - self.enemy[k]).collect();
        let mut e_can_capture = vec![0i64; n];
        for k in 0..n {
            if ebrdr[k] != 0 && self.nb[k].iter().any(|&m| self.enemy[m] != 0 && self.strn[m] > self.strn[k]) {
                e_can_capture[k] = 1;
            }
        }

        let target_cells: Vec<i64> = (0..n).map(|k| self.enemy[k] + self.blank[k] * (self.strn[k] == 0) as i64).collect();
        let ubrdr_combat: Vec<i64> = (0..n).map(|k| self.ubrdr[k] * target_cells[k]).collect();
        let pc = self.plus(&ubrdr_combat, max);
        self.melee = (0..n).map(|k| pc[k] * self.owned[k]).collect();
        let mut close = self.melee.clone();
        for _ in 0..COM_RADIUS {
            let p = self.plus(&close, max);
            close = (0..n).map(|k| close[k].max(p[k])).collect();
        }
        for k in 0..n {
            close[k] = (close[k] - self.melee[k]) * self.owned[k];
        }
        if target_cells.iter().any(|&t| t > 0) {
            let src: Vec<usize> = (0..n).filter(|&k| ubrdr_combat[k] != 0).collect();
            self.dfc = self.friendly_to(&src);
        } else {
            self.dfc = vec![0; n];
        }
        for k in 0..n {
            if self.dfc[k] > COM_RADIUS {
                close[k] = 0;
            }
        }
        self.close = close;

        let targets: Vec<i64> = (0..n)
            .map(|k| {
                let t = self.strn[k] * self.blank[k];
                if t == 0 { 256 } else { t }
            })
            .collect();
        let weakest = self.plus(&targets, min);
        self.gte_nbr = (0..n).map(|k| self.strn[k] * self.owned[k] > weakest[k]).collect();

        self.calc_bval();
        self.calc_aggs(&ebrdr, &e_can_capture);
    }

    fn calc_aggs(&mut self, ebrdr: &[i64], e_can_capture: &[i64]) {
        let n = self.n;
        let ostrn: i64 = (0..n).map(|k| self.strn[k] * self.owned[k]).sum();
        let oprod: i64 = (0..n).map(|k| self.prod[k] * self.owned[k]).sum();
        self.total_strn = ostrn as f64 + oprod as f64 / 2.0;
        let mut seen = [false; 256];
        for &o in &self.owners {
            seen[o as usize] = true;
        }
        let uniq = seen.iter().filter(|&&s| s).count() as i64;
        self.num_enemies = 1.max(uniq - 2);
        let total_enemy: i64 = (0..n).map(|k| self.strn[k] * self.enemy[k]).sum();
        let ave_enemy = total_enemy as f64 / self.num_enemies as f64;
        self.in_combat = self.melee.iter().any(|&m| m > 0);

        let max = |a: i64, b: i64| a.max(b);
        let pe = self.plus(&self.enemy, max);
        let mut walls: Vec<i64> = (0..n).map(|k| self.blank[k] * (self.strn[k] > 0) as i64 * pe[k]).collect();
        if !self.in_combat && self.total_strn < 2.0 * ave_enemy && (self.turn - 5) < self.last_turn {
            let pcc = self.plus(e_can_capture, max);
            for k in 0..n {
                walls[k] = (walls[k] + pcc[k] - self.owned[k]).min(1);
            }
        }
        self.safe = walls.iter().map(|&e| 1 - e).collect();

        if !self.in_combat && self.total_strn > 2.0 * ave_enemy {
            let min_str = (0..n).filter(|&k| walls[k] != 0).map(|k| self.strn[k]).min();
            if let Some(ms) = min_str {
                for k in 0..n {
                    if self.strn[k] == ms {
                        walls[k] = 0;
                        self.safe[k] = 1;
                    }
                }
            }
        }
        if self.in_combat && self.num_enemies == 1 {
            let min_str = (0..n).filter(|&k| ebrdr[k] != 0 && self.obrdr[k] != 0).map(|k| self.strn[k]).min();
            if let Some(ms) = min_str {
                for k in 0..n {
                    if self.strn[k] == ms {
                        self.safe[k] = 1;
                    }
                }
            }
        }
    }

    /// The tunnel value T of every unowned border site, from the init-time shortest paths.
    fn calc_bval(&mut self) {
        let n = self.n;
        let bis: Vec<usize> = (0..n).filter(|&k| self.ubrdr[k] != 0).collect();
        let mut bvals = vec![0f64; bis.len()];
        for u in 0..n {
            if !(self.blank[u] != 0 && self.strn[u] > 0) {
                continue;
            }
            let row = |b: usize| self.path[b * n + u];
            let mn = bis.iter().map(|&b| row(b)).min();
            let mn = match mn {
                Some(m) if m != INF => m,
                _ => continue,
            };
            let (up, us) = (self.prodc[u] as f64, self.strnc[u] as f64);
            let d = mn as f64 + us / up + 1.0;
            let v = (up * up / us) / d;
            for (i, &b) in bis.iter().enumerate() {
                if row(b) == mn {
                    bvals[i] += v;
                }
            }
        }
        self.mbval = vec![0f64; n];
        for (i, &b) in bis.iter().enumerate() {
            self.mbval[b] += bvals[i];
        }
    }

    // ---- Combatant ----

    fn decide_combat_moves(&mut self, ms: &mut Moveset) {
        let n = self.n;
        let locs: Vec<usize> = (0..n).filter(|&k| self.melee[k] != 0).collect();
        let strns: Vec<i64> = locs.iter().map(|&k| self.strn[k]).collect();
        for ci in argsort_desc(&strns) {
            let c = locs[ci];
            if self.strnc[c] < self.prodc[c] * COMBAT_WAIT {
                continue;
            }
            let mut best = 0;
            let mut best_score = i64::MIN;
            for (i, &m) in self.nb[c].iter().enumerate() {
                let s = self.combat_heur[m] - self.wall[m] * 100000;
                if s > best_score {
                    best_score = s;
                    best = i;
                }
            }
            let mut t = self.nb[c][best];
            if self.total_strn < self.strn[t] as f64 || self.safe[t] == 0 {
                t = c;
            }
            // numpy scalar: int(x / 10000)
            self.combat_heur[t] = (self.combat_heur[t] as f64 / 10000.0) as i64;
            ms.add_move(c, t, None);
        }

        for c in 0..n {
            if self.close[c] == 0 {
                continue;
            }
            if self.strnc[c] < self.prodc[c] * COMBAT_WAIT {
                continue;
            }
            if self.dfc[c] > 6 {
                continue;
            }
            let mut t = 0;
            let mut bv = f64::NEG_INFINITY;
            for k in 0..n {
                let v = self.melee[k] as f64 / self.dist(c, k) as f64;
                if v > bv {
                    bv = v;
                    t = k;
                }
            }
            if self.total_strn < self.strn[t] as f64 || self.safe[t] == 0 {
                t = c;
            }
            if self.dist(c, t) < 4 && (self.strnc[c] as f64) < self.prodc[c] as f64 * (COMBAT_WAIT as f64 + 1.5) {
                ms.add_move(c, c, None);
            } else {
                ms.add_move(c, t, None);
            }
        }
    }

    // ---- MoveMaker ----

    fn decide_noncombat_moves(&mut self, ms: &mut Moveset) {
        let n = self.n;
        let working_strn = self.strn.clone();
        let mut motile: Vec<bool> = (0..n).map(|k| self.strnc[k] >= self.prodc[k] * WAIT && self.owned[k] != 0).collect();
        for k in 0..n {
            if self.gte_nbr[k] {
                motile[k] = true;
            }
            if ms.move_matrix[k] {
                motile[k] = false;
            }
        }

        let mut vmid = vec![0f64; n];
        let mut vglob = vec![0f64; n];
        let mut vtot_max = 0f64;
        for k in 0..n {
            if self.ubrdr[k] == 0 {
                continue;
            }
            let s = self.safe[k] as f64;
            let mid = self.gauss_mid[k];
            let local = mid.max((self.prodc[k] * self.prodc[k]) as f64 / self.strnc[k] as f64);
            vmid[k] = mid * s;
            vglob[k] = self.mbval[k] * self.glob_k * s;
            let tot = local * s + vmid[k] + vglob[k];
            if tot > vtot_max {
                vtot_max = tot;
            }
        }
        if vtot_max == 0.0 {
            self.process_wallup(ms);
            return;
        }
        let border: Vec<usize> = (0..n).filter(|&k| self.ubrdr[k] != 0).collect();

        let mut desired: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut desired_idx: Vec<Option<usize>> = vec![None; n];
        let mut d1_conquered = vec![1f64; n];

        let locs: Vec<usize> = (0..n).filter(|&k| motile[k]).collect();
        let strns: Vec<i64> = locs.iter().map(|&k| self.strn[k]).collect();
        for ai in argsort_desc(&strns) {
            let a = locs[ai];
            let sa = self.strn[a];
            // argmax over the whole map: values are 0 off the border, so the first border site
            // with the largest positive value, else site 0
            let mut t = 0;
            let mut bv = 0f64;
            for &k in &border {
                let d = self.dist(a, k) as f64;
                let t2c = (0f64).max((working_strn[k] - sa) as f64 / self.prodc[k] as f64);
                let mut v = vmid[k] / (d + t2c) * d1_conquered[k] + vglob[k] / (d + BULK_MVMT_OFF);
                if k == a {
                    v *= 2.0;
                }
                if v > bv {
                    bv = v;
                    t = k;
                }
            }
            if self.total_strn < self.strn[t] as f64 || self.safe[t] == 0 {
                t = a;
            }
            ms.add_move(a, t, None);
            if self.dist(a, t) == 1 {
                let di = match desired_idx[t] {
                    Some(i) => i,
                    None => {
                        desired.push((t, Vec::new()));
                        desired_idx[t] = Some(desired.len() - 1);
                        desired.len() - 1
                    }
                };
                desired[di].1.push(a);
                if self.strn[a] > self.strn[t] {
                    d1_conquered[t] = 0.0;
                }
            }
        }

        // process_d1_teamups: let two weak pieces jointly take a site
        for (t, locs) in &desired {
            if locs.len() < 2 {
                continue;
            }
            let total: i64 = locs.iter().map(|&a| self.strn[a]).sum();
            let mx = locs.iter().map(|&a| self.strn[a]).max().unwrap();
            if total > self.strn[*t] && mx <= self.strn[*t] {
                self.strn[*t] = 0;
            }
        }
    }

    fn process_wallup(&mut self, ms: &mut Moveset) {
        if self.turn == self.last_turn {
            self.process_final_turn(ms);
            return;
        }
        let src: Vec<usize> = (0..self.n).filter(|&k| self.obrdr[k] != 0).collect();
        let d2w = self.friendly_to(&src);
        for a in 0..self.n {
            if self.owned[a] == 0 {
                continue;
            }
            if self.strnc[a] < self.prodc[a] * WAIT {
                continue;
            }
            let opts: Vec<usize> = self.nb[a].iter().copied().filter(|&m| d2w[m] < d2w[a] && self.strn[m] < self.strn[a]).collect();
            if opts.is_empty() {
                ms.add_move(a, a, None);
            } else {
                let m = opts[self.rng.below(opts.len())];
                ms.add_move(a, m, None);
            }
        }
    }

    fn process_final_turn(&mut self, ms: &mut Moveset) {
        for s in self.safe.iter_mut() {
            *s = 1;
        }
        for a in 0..self.n {
            if self.obrdr[a] == 0 {
                continue;
            }
            let opts: Vec<usize> = self.nb[a].iter().copied().filter(|&m| self.owned[m] == 0 && self.strn[a] > self.strn[m]).collect();
            if opts.is_empty() {
                continue;
            }
            let m = opts[self.rng.below(opts.len())];
            ms.add_move(a, m, None);
        }
    }

    // ---- Resolver ----

    fn step(&self, k: usize, card: Option<u8>) -> usize {
        match card {
            Some(d) if d >= 1 => self.nb[k][d as usize - 1],
            _ => k,
        }
    }

    fn can_occupy_safely(&self, a: usize, m: usize) -> bool {
        if self.owned[m] != 0 && self.strn[m] > self.strn[a] && self.strn[m] + self.strn[a] > 255 {
            return false;
        }
        if self.owned[m] != 0 {
            return true;
        }
        if self.safe[m] == 0 {
            return false;
        }
        if self.strn[a] >= 255 {
            return true;
        }
        self.strn[m] < self.strn[a]
    }

    /// find_pref_next: up to two next steps toward t, in order of preference.
    fn find_pref_next(&self, a: usize, t: usize) -> ((usize, u8), Option<(usize, u8)>) {
        let (x, y) = (self.xs[a] as i64, self.ys[a] as i64);
        let (nx, ny) = (self.xs[t] as i64, self.ys[t] as i64);
        let (w, h) = (self.w as i64, self.h as i64);
        let dn = (y - ny).rem_euclid(h);
        let de = (nx - x).rem_euclid(w);
        let ds = (ny - y).rem_euclid(h);
        let dw = (x - nx).rem_euclid(w);
        let (ypref, ydist) = if dn < ds {
            (Some(NORTH), dn)
        } else if dn == 0 {
            (None, 0)
        } else {
            (Some(SOUTH), ds)
        };
        let (xpref, xdist) = if de < dw {
            (Some(EAST), de)
        } else if de == 0 {
            (None, 0)
        } else {
            (Some(WEST), dw)
        };
        let xn = self.step(a, xpref);
        let yn = self.step(a, ypref);
        let xo = (xn, xpref.unwrap_or(0));
        let yo = (yn, ypref.unwrap_or(0));

        if self.owned[xn] != 0 && self.owned[yn] != 0 && ydist > 0 && xdist > 0 {
            return if self.prod[xn] < self.prod[yn] { (xo, Some(yo)) } else { (yo, Some(xo)) };
        }
        let can_x = self.can_occupy_safely(a, xn) && xdist > 0;
        let can_y = self.can_occupy_safely(a, yn) && ydist > 0;
        if self.owned[xn] != 0 && xdist > 0 && can_y {
            return (xo, Some(yo));
        }
        if self.owned[yn] != 0 && ydist > 0 && can_x {
            return (yo, Some(xo));
        }
        if self.owned[xn] != 0 && xdist > 0 {
            return (xo, None);
        }
        if self.owned[yn] != 0 && ydist > 0 {
            return (yo, None);
        }
        if can_x && can_y {
            if self.enemy[xn] != 0 {
                return (xo, Some(yo));
            }
            if self.enemy[yn] != 0 {
                return (yo, Some(xo));
            }
            let x_roi = self.prodc[xn] as f64 / self.strnc[xn] as f64;
            let y_roi = self.prodc[yn] as f64 / self.strnc[yn] as f64;
            return if x_roi > y_roi { (xo, Some(yo)) } else { (yo, Some(xo)) };
        }
        if can_x {
            return (xo, None);
        }
        if can_y {
            return (yo, None);
        }
        ((a, STILL), None)
    }

    fn resolve(&mut self, ms: &mut Moveset) {
        ms.set_stays();
        let n = self.n;
        let mut pstrn = vec![0i64; n];
        let all_on = (self.turn < 40 && !self.in_combat) || self.turn == self.last_turn;
        let mut on: Vec<(usize, usize)> = Vec::new();
        let mut off: Vec<usize> = Vec::new();
        for &a in &ms.order {
            let (t, _) = ms.entry[a].unwrap();
            if all_on || (self.xs[a] as i64 + self.ys[a] as i64 + self.turn) % 2 == 0 {
                on.push((a, t));
            } else {
                off.push(a);
            }
        }
        let on_strns: Vec<i64> = on.iter().map(|&(a, _)| self.strn[a]).collect();
        for i in argsort_desc(&on_strns) {
            let (a, t) = on[i];
            let istrn = on_strns[i];
            let strlim_cell = istrn.max(STRLIM);
            let (p1, p2) = if self.close[a] != 0 && a != t {
                let d2c = self.dfc[a];
                let mut choices: Vec<(usize, u8)> = Vec::new();
                let mut cdists: Vec<f64> = Vec::new();
                for (j, &m) in self.nb[a].iter().enumerate() {
                    if self.dfc[m] < d2c && self.wall[m] == 0 {
                        choices.push((m, j as u8 + 1));
                        cdists.push(self.dist(t, m) as f64 + self.prod[m] as f64 * 0.01);
                    }
                }
                if choices.len() == 2 {
                    if cdists[1] < cdists[0] {
                        (choices[1], Some(choices[0]))
                    } else {
                        (choices[0], Some(choices[1]))
                    }
                } else if !choices.is_empty() {
                    (choices[0], None)
                } else {
                    ((a, STILL), None)
                }
            } else {
                self.find_pref_next(a, t)
            };

            if istrn + pstrn[p1.0] <= strlim_cell {
                ms.add_move(a, p1.0, Some(p1.1));
                pstrn[p1.0] += istrn;
            } else if p2.is_some() && istrn + pstrn[p2.unwrap().0] <= strlim_cell {
                let p2 = p2.unwrap();
                ms.add_move(a, p2.0, Some(p2.1));
                pstrn[p2.0] += istrn;
            } else if self.melee[a] != 0 {
                let mut best = 0;
                let mut bs = i64::MIN;
                for (j, &m) in self.nb[a].iter().enumerate() {
                    let s = self.combat_heur[m] * ((pstrn[m] + istrn) < strlim_cell) as i64;
                    if s > bs {
                        bs = s;
                        best = j;
                    }
                }
                if bs > 0 {
                    let m = self.nb[a][best];
                    ms.add_move(a, a, Some(best as u8 + 1));
                    pstrn[m] += istrn + self.prod[a];
                } else {
                    ms.add_move(a, a, Some(STILL));
                    pstrn[a] += istrn + self.prod[a];
                }
            } else {
                off.push(a);
            }
        }

        // the off squares get out of the way
        for &a in &off {
            let istrn = self.strnc[a];
            let iprod = self.prod[a];
            let strlim_cell = istrn.max(STRLIM);
            if pstrn[a] == 0 || pstrn[a] + istrn + iprod <= strlim_cell {
                ms.add_move(a, a, Some(STILL));
                pstrn[a] += istrn + iprod;
                continue;
            }
            let addable = 255 - pstrn[a] - istrn;
            if addable > istrn {
                ms.add_move(a, a, Some(STILL));
                pstrn[a] += istrn + iprod;
                continue;
            }
            let nbrs = self.nb[a];
            let argmax = |v: [i64; 4]| (0..4).fold(0, |b, j| if v[j] > v[b] { j } else { b });
            let es: [i64; 4] = std::array::from_fn(|j| self.enemy[nbrs[j]] * self.strn[nbrs[j]]);
            let j = argmax(es);
            if es[j] > 1 {
                ms.add_move(a, nbrs[j], Some(j as u8 + 1));
                pstrn[nbrs[j]] += istrn;
                continue;
            }
            let bs: [i64; 4] = std::array::from_fn(|j| {
                let m = nbrs[j];
                self.blank[m] * self.prod[m] * (self.strnc[m] < istrn) as i64 * self.safe[m]
            });
            let j = argmax(bs);
            if bs[j] as f64 > 0.5 {
                ms.add_move(a, nbrs[j], Some(j as u8 + 1));
                pstrn[nbrs[j]] += istrn;
                continue;
            }
            let os: [i64; 4] = std::array::from_fn(|j| {
                let m = nbrs[j];
                let v = self.owned[m] * (pstrn[m] + self.strn[m]);
                if v == 0 { 999 } else { v }
            });
            let j = (0..4).fold(0, |b, j| if os[j] < os[b] { j } else { b });
            ms.add_move(a, nbrs[j], Some(j as u8 + 1));
            pstrn[nbrs[j]] += istrn;
        }
    }

    // ---- Nonaggressor ----

    fn nonaggress(&self, ms: &mut Moveset) {
        let stays: Vec<usize> = ms.order.iter().copied().filter(|&a| self.safe[ms.entry[a].unwrap().0] == 0).collect();
        for a in stays {
            ms.add_move(a, a, Some(STILL));
        }
    }

    fn turn_moves(&mut self) -> Vec<Move> {
        self.update();
        let mut ms = Moveset::new(&self.owned);
        self.decide_combat_moves(&mut ms);
        self.decide_noncombat_moves(&mut ms);
        self.resolve(&mut ms);
        self.nonaggress(&mut ms);
        ms.order
            .iter()
            .map(|&a| {
                let (_, d) = ms.entry[a].unwrap();
                Move { loc: Location { x: self.xs[a] as u16, y: self.ys[a] as u16 }, direction: d.unwrap_or(STILL) }
            })
            .collect()
    }
}

/// scipy.sparse.csgraph.dijkstra(dok, directed=False) on the init map: edge i->j costs
/// int(strength[j] / (production[i] + 0.1)), zero costs are not stored (no edge), and an
/// undirected search may use either direction of an edge.
fn all_pairs(n: usize, nb: &[[usize; 4]], strn: &[i64], prod: &[i64]) -> Vec<u32> {
    let cost = |i: usize, j: usize| (strn[j] as f64 / (prod[i] as f64 + 0.1)) as u32;
    let mut adj: Vec<Vec<(usize, u32)>> = vec![Vec::new(); n];
    for i in 0..n {
        for &j in &nb[i] {
            let mut best = INF;
            for c in [cost(i, j), cost(j, i)] {
                if c != 0 && c < best {
                    best = c;
                }
            }
            if best != INF && !adj[i].iter().any(|&(m, _)| m == j) {
                adj[i].push((j, best));
            }
        }
    }
    let mut path = vec![INF; n * n];
    let mut heap = BinaryHeap::new();
    for s in 0..n {
        let row = &mut path[s * n..(s + 1) * n];
        row[s] = 0;
        heap.push(Reverse((0u32, s)));
        while let Some(Reverse((d, u))) = heap.pop() {
            if d > row[u] {
                continue;
            }
            for &(v, c) in &adj[u] {
                let nd = d + c;
                if nd < row[v] {
                    row[v] = nd;
                    heap.push(Reverse((nd, v)));
                }
            }
        }
    }
    path
}

/// scipy.ndimage.gaussian_filter(v, sigma, mode='wrap') on an [x, y] array.
fn gaussian_wrap(v: &[f64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    let radius = (4.0 * sigma + 0.5) as i64;
    let mut kern: Vec<f64> = (-radius..=radius).map(|x| (-0.5 / (sigma * sigma) * (x * x) as f64).exp()).collect();
    let s: f64 = kern.iter().sum();
    for k in kern.iter_mut() {
        *k /= s;
    }
    let mut a = vec![0f64; w * h];
    for x in 0..w {
        for y in 0..h {
            let mut acc = 0.0;
            for (i, &kv) in kern.iter().enumerate() {
                let xx = (x as i64 + i as i64 - radius).rem_euclid(w as i64) as usize;
                acc += v[xx * h + y] * kv;
            }
            a[x * h + y] = acc;
        }
    }
    let mut b = vec![0f64; w * h];
    for x in 0..w {
        for y in 0..h {
            let mut acc = 0.0;
            for (i, &kv) in kern.iter().enumerate() {
                let yy = (y as i64 + i as i64 - radius).rem_euclid(h as i64) as usize;
                acc += a[x * h + yy] * kv;
            }
            b[x * h + y] = acc;
        }
    }
    b
}

fn main() {
    let (me, init_map, mut conn) = get_init();
    let (w, h) = (init_map.width as usize, init_map.height as usize);
    let n = w * h;
    let mut xs = vec![0; n];
    let mut ys = vec![0; n];
    let mut nb = vec![[0usize; 4]; n];
    let mut prod = vec![0i64; n];
    let mut strn0 = vec![0i64; n];
    let mut seed = 0x2545_f491_4f6c_dd1du64 ^ ((me as u64) << 40);
    for x in 0..w {
        for y in 0..h {
            let k = x * h + y;
            xs[k] = x;
            ys[k] = y;
            nb[k] = [
                x * h + (y + h - 1) % h,
                ((x + 1) % w) * h + y,
                x * h + (y + 1) % h,
                ((x + w - 1) % w) * h + y,
            ];
            let s = init_map.contents[y][x];
            prod[k] = s.production as i64;
            strn0[k] = s.strength as i64;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(s.production as u64 * 256 + s.strength as u64 + 1);
        }
    }
    let path = all_pairs(n, &nb, &strn0, &prod);
    let prodc: Vec<i64> = prod.iter().map(|&p| p.max(1)).collect();
    // original_strn is never 0 on this engine's maps; the floor only avoids a division by zero
    let mid_raw: Vec<f64> = (0..n).map(|k| (prodc[k] * prodc[k]) as f64 / strn0[k].max(1) as f64).collect();
    let gauss_mid = gaussian_wrap(&mid_raw, w, h, 1.2);
    send_init("DexBot");

    let mut bot = Dex {
        w,
        h,
        n,
        me,
        turn: -1,
        last_turn: (((w * h) as f64).sqrt() * 10.0).floor() as i64,
        glob_k: 0.0,
        nb,
        xs,
        ys,
        prod,
        prodc,
        path,
        gauss_mid,
        rng: Rng(seed | 1),
        owners: vec![0; n],
        strn: vec![0; n],
        owned: vec![0; n],
        blank: vec![0; n],
        enemy: vec![0; n],
        strnc: vec![0; n],
        wall: vec![0; n],
        combat_heur: vec![0; n],
        ubrdr: vec![0; n],
        obrdr: vec![0; n],
        melee: vec![0; n],
        close: vec![0; n],
        dfc: vec![0; n],
        gte_nbr: vec![false; n],
        mbval: vec![0.0; n],
        total_strn: 0.0,
        num_enemies: 1,
        in_combat: false,
        safe: vec![1; n],
    };

    // The original calls update() once on the first frame before its loop (turn 0), which
    // fixes k = 1.5 - 0.1 * number of enemies, then again inside the loop (turn 1).
    bot.read_frame(&conn.get_frame());
    bot.update();
    bot.glob_k = 1.5 - bot.num_enemies as f64 * 0.1;
    loop {
        let moves = bot.turn_moves();
        conn.send_frame(&moves);
        bot.read_frame(&conn.get_frame());
    }
}
