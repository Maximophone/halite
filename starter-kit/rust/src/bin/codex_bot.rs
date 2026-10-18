//! CodexBot: an empirically tuned challenger derived from the repository's ErdmanBot.
//!
//! Attribution: Travis Erdman's 2016 winning strategy, ported in `erdman_bot.rs`.
//! Source: https://github.com/erdman/erdman-halite-bots (`erdman_v26.py`).
//! Retains the potential-field economy, synchronized gathering trees, strength-cap
//! control and local combat avoidance. Experiments tune the wall and waiting rules
//! independently; all CODEX_* settings are read once at initialization.
//! See `tools/codex_experiments/` for the reproducible paired-seat benchmark.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use hlt::*;

// Read experiment settings once, never in the per-square hot path.
struct Config {
    alpha: f64,
    degradation: f64,
    enemy_roi: f64,
    combat_hold: i64,
    int_max: f64,
    int_min: f64,
    mining_hold: i64,
    danger: f64,
    strategic: bool,
    walls: bool,
    prod_power: f64,
    breach: f64,
    auto: bool,
    duel_hold: i64,
    duel_int_max: f64,
}
fn config() -> &'static Config {
    static CONFIG: std::sync::OnceLock<Config> = std::sync::OnceLock::new();
    CONFIG.get_or_init(|| {
        let get = |key: &str, default: f64| {
            std::env::var(format!("CODEX_{key}"))
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| v.is_finite())
                .unwrap_or(default)
        };
        Config {
            alpha: get("ALPHA", 0.1),
            degradation: get("DEGRADE", 0.4),
            enemy_roi: get("ENEMY", -0.5),
            combat_hold: get("HOLD", 7.0) as i64,
            int_max: get("INT_MAX", 0.45),
            int_min: get("INT_MIN", 0.01),
            mining_hold: get("MINE_HOLD", 5.0) as i64,
            danger: get("DANGER", 5000.0),
            strategic: get("STRATEGIC", 1.0) != 0.0,
            walls: get("WALLS", 1.0) != 0.0,
            prod_power: get("PROD_POWER", 1.0),
            breach: get("BREACH", 0.0),
            auto: get("AUTO", 1.0) != 0.0,
            duel_hold: get("DUEL_HOLD", 8.0) as i64,
            duel_int_max: get("DUEL_INT_MAX", 0.75),
        }
    })
}
// Python kit directions: NORTH, EAST, SOUTH, WEST, STILL = 0..4
const PSTILL: usize = 4;

/// CPython's `random.Random` (MT19937, `init_by_array` seeding, 53-bit `random()`).
struct PyRandom {
    mt: [u32; 624],
    index: usize,
}

impl PyRandom {
    fn from_key(key: &[u32]) -> PyRandom {
        const N: usize = 624;
        let key: Vec<u32> = if key.is_empty() {
            vec![0]
        } else {
            key.to_vec()
        };
        let mut mt = [0u32; N];
        mt[0] = 19650218;
        for i in 1..N {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            let prev = mt[i - 1] ^ (mt[i - 1] >> 30);
            mt[i] = (mt[i] ^ prev.wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let prev = mt[i - 1] ^ (mt[i - 1] >> 30);
            mt[i] = (mt[i] ^ prev.wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
        }
        mt[0] = 0x8000_0000;
        PyRandom { mt, index: N }
    }

    /// `random.seed(n)` for an int n: the key is abs(n) as 32-bit words, least significant first.
    fn from_int(n: i128) -> PyRandom {
        let mut v = n.unsigned_abs();
        let mut key = Vec::new();
        while v > 0 {
            key.push(v as u32);
            v >>= 32;
        }
        PyRandom::from_key(&key)
    }

    fn genrand_u32(&mut self) -> u32 {
        const N: usize = 624;
        const M: usize = 397;
        if self.index >= N {
            for kk in 0..N {
                let y = (self.mt[kk] & 0x8000_0000) | (self.mt[(kk + 1) % N] & 0x7fff_ffff);
                let mag = if y & 1 != 0 { 0x9908_b0df } else { 0 };
                self.mt[kk] = self.mt[(kk + M) % N] ^ (y >> 1) ^ mag;
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }
}

/// Python tuple order on (value, random): the random draw breaks value ties.
fn tuple_cmp(a: (f64, f64), b: (f64, f64)) -> Ordering {
    a.0.partial_cmp(&b.0)
        .unwrap_or(Ordering::Equal)
        .then(a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
}

/// heapq entry `(priority, random, potential, friendly_distance, square)`, popped smallest first.
struct Entry {
    prio: f64,
    rnd: f64,
    pot: f64,
    fd: i64,
    sq: usize,
}

impl PartialEq for Entry {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Entry {
    fn cmp(&self, o: &Self) -> Ordering {
        // reversed: BinaryHeap is a max-heap
        tuple_cmp((o.prio, o.rnd), (self.prio, self.rnd))
    }
}

fn degrade_potential(potential: f64, distance: i64, step: f64) -> f64 {
    potential + step * (distance * distance) as f64
}

struct Bot {
    cfg: &'static Config,
    duel: bool,
    production_power: f64,
    degradation: f64,
    n: usize,
    me: u8,
    xs: Vec<usize>,
    ys: Vec<usize>,
    /// N, E, S, W, self
    nb: Vec<[usize; 5]>,
    /// distance-2 diamond without self, in hlt's order (dy outer, dx inner)
    nb2: Vec<Vec<usize>>,
    prod: Vec<i64>,
    rng: PyRandom,
    // per turn
    owner: Vec<u8>,
    strn: Vec<i64>,
    wall: Vec<bool>,
    mining_remains: bool,
    pf: Vec<(f64, i64)>,
    dest: Vec<i64>,
    redlight: Vec<bool>,
    greenlight: Vec<bool>,
    hold_until: i64,
    strength_hurdle: i64,
}

impl Bot {
    fn mine(&self, i: usize) -> bool {
        self.owner[i] == self.me
    }

    fn is_enemy(&self, i: usize) -> bool {
        self.owner[i] != 0 && self.owner[i] != self.me
    }

    fn is_empty(&self, i: usize) -> bool {
        self.owner[i] == 0 && self.strn[i] == 0
    }

    fn pfv(&self, i: usize) -> f64 {
        degrade_potential(self.pf[i].0, self.pf[i].1, self.degradation)
    }

    fn initial_potential(&self, i: usize) -> f64 {
        if self.is_empty(i) {
            let mut s = 0.0;
            for &m in &self.nb[i][..4] {
                if self.is_enemy(m) {
                    s += self.cfg.enemy_roi;
                }
            }
            s
        } else if self.prod[i] == 0 || self.is_enemy(i) {
            f64::INFINITY
        } else if self.wall[i] {
            (100 * self.strn[i]) as f64 / (self.prod[i] as f64).powf(self.production_power)
        } else {
            self.strn[i] as f64 / (self.prod[i] as f64).powf(self.production_power)
        }
    }

    fn best_neighbor(&self, i: usize) -> usize {
        let mut best = self.nb[i][0];
        for &m in &self.nb[i][1..4] {
            if self.pfv(m) < self.pfv(best) {
                best = m;
            }
        }
        best
    }

    fn potential_field(&mut self) {
        let n = self.n;
        let mut heap = BinaryHeap::new();
        for i in 0..n {
            if !self.mine(i) {
                let p = self.initial_potential(i);
                let rnd = self.rng.random();
                heap.push(Entry {
                    prio: p,
                    rnd,
                    pot: p,
                    fd: 0,
                    sq: i,
                });
            }
        }
        let mut done = vec![false; n];
        let mut count = 0;
        while count < n {
            let e = heap.pop().expect("frontier exhausted");
            if done[e.sq] {
                continue;
            }
            done[e.sq] = true;
            count += 1;
            self.pf[e.sq] = (e.pot, e.fd);
            for d in 0..4 {
                let m = self.nb[e.sq][d];
                if self.wall[m] {
                    continue; // don't carve a path through the wall, go around it
                } else if !self.mine(m) {
                    let np = if self.prod[m] != 0 && self.owner[m] == 0 {
                        (1.0 - self.cfg.alpha) * e.pot
                            + self.cfg.alpha * self.strn[m] as f64
                                / (self.prod[m] as f64).powf(self.production_power)
                    } else {
                        f64::INFINITY
                    };
                    let rnd = self.rng.random();
                    heap.push(Entry {
                        prio: np,
                        rnd,
                        pot: np,
                        fd: e.fd,
                        sq: m,
                    });
                } else {
                    let np = degrade_potential(e.pot, e.fd + 1, self.degradation);
                    let rnd = self.rng.random();
                    heap.push(Entry {
                        prio: np,
                        rnd,
                        pot: e.pot,
                        fd: e.fd + 1,
                        sq: m,
                    });
                }
            }
        }
    }

    fn red_green(&mut self) {
        let n = self.n;
        let mut kids: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut is_parent = vec![false; n];
        for i in 0..n {
            if self.mine(i) {
                let p = self.best_neighbor(i);
                kids[p].push(i);
                is_parent[p] = true;
            }
        }
        for v in self.redlight.iter_mut() {
            *v = false;
        }
        for v in self.greenlight.iter_mut() {
            *v = false;
        }
        for root in 0..n {
            // roots: parents that are nobody's child (children are all ours), neutral and not empty
            if !is_parent[root]
                || self.mine(root)
                || !(self.owner[root] == 0 && self.strn[root] > 0)
            {
                continue;
            }
            let mut nodes: Vec<(usize, usize)> = Vec::new();
            let mut stack: Vec<(usize, usize)> = kids[root].iter().map(|&k| (1, k)).collect();
            while let Some((lvl, k)) = stack.pop() {
                nodes.push((lvl, k));
                for &c in &kids[k] {
                    stack.push((lvl + 1, c));
                }
            }
            // sorted(walk_tree(tree)): by level, then Square = (x, y, ...)
            nodes.sort_by_key(|&(lvl, k)| (lvl, self.xs[k], self.ys[k]));
            let root_s = self.strn[root];
            let (mut acc_s, mut acc_p) = (0i64, 0i64);
            let mut idx = 0;
            while idx < nodes.len() {
                let lvl = nodes[idx].0;
                let mut squares: Vec<usize> = Vec::new();
                while idx < nodes.len() && nodes[idx].0 == lvl {
                    squares.push(nodes[idx].1);
                    idx += 1;
                }
                let mut ls: i64 = squares.iter().map(|&k| self.strn[k]).sum();
                let lp: i64 = squares.iter().map(|&k| self.prod[k]).sum();
                if acc_s + acc_p > root_s {
                    break;
                } else if ls + acc_s + acc_p > root_s {
                    // stable sort by strength, descending (Python's reverse=True keeps tie order)
                    squares.sort_by(|&a, &b| self.strn[b].cmp(&self.strn[a]));
                    while ls + acc_s + acc_p - self.strn[*squares.last().unwrap()] > root_s {
                        ls -= self.strn[squares.pop().unwrap()];
                    }
                    for &k in &squares {
                        self.greenlight[k] = true;
                    }
                    break;
                } else {
                    acc_s += ls + acc_p;
                    acc_p += lp;
                    for &k in &squares {
                        self.redlight[k] = true;
                    }
                }
            }
        }
    }

    /// Returns the move in the Python kit's numbering (NORTH=0 .. WEST=3, STILL=4).
    fn assign_move(&mut self, sq: usize) -> usize {
        let s = self.strn[sq];
        let p = self.prod[sq];
        let mut best: Option<((f64, f64), usize)> = None;
        for d in 0..4 {
            let m = self.nb[sq][d];
            let v = self.pfv(m)
                + (10000 * (self.dest[m] + s - 255).max(0)) as f64
                + if self.mining_remains && self.wall[m] {
                    1e7
                } else {
                    0.0
                };
            let r = self.rng.random();
            if best.map_or(true, |(b, _)| tuple_cmp((v, r), b) == Ordering::Less) {
                best = Some(((v, r), d));
            }
        }
        let ((potential, _), best_d) = best.unwrap();
        let target = self.nb[sq][best_d];

        if self.greenlight[sq] {
            return best_d;
        }

        let stay_loss = ((s + p).min(255) + self.dest[sq] - 255).max(0);

        if stay_loss == 0 && (self.dest[sq] > 0 || self.redlight[sq]) {
            return PSTILL; // safely meld with all oncoming
        }

        // dangerous empties: combat empties (or enemy squares) within 2 that touch a live enemy
        // and that one of our moves already lands on or next to
        let mut dangerous: Vec<(usize, i64)> = Vec::new();
        for &m in &self.nb2[sq] {
            if dangerous.iter().any(|&(k, _)| k == m) {
                continue;
            }
            let around = &self.nb[m];
            if (self.is_empty(m) || self.is_enemy(m))
                && around.iter().any(|&k| self.is_enemy(k) && self.strn[k] > 0)
                && around.iter().any(|&k| self.dest[k] > 0)
            {
                let tot: i64 = around
                    .iter()
                    .filter(|&&k| self.is_enemy(k))
                    .map(|&k| self.strn[k])
                    .sum();
                dangerous.push((m, tot.min(255)));
            }
        }
        let is_dangerous = |k: usize| dangerous.iter().any(|&(d, _)| d == k);

        if self.cfg.strategic
            && !(self.cfg.auto && self.duel)
            && stay_loss == 0
            && dangerous.is_empty()
        {
            let empties: Vec<usize> = self.nb[sq][..4]
                .iter()
                .copied()
                .filter(|&m| self.is_empty(m))
                .collect();
            if empties.len() > 1 {
                let mut threats: Vec<usize> = Vec::new();
                for &e in &empties {
                    for &k in &self.nb[e][..4] {
                        if self.is_enemy(k)
                            && self.strn[k] >= 3 * self.prod[k]
                            && !threats.contains(&k)
                        {
                            threats.push(k);
                        }
                    }
                }
                if threats.len() > 1 {
                    return PSTILL; // strategic stilling
                }
            }
        }

        if self.nb[target].iter().any(|&k| is_dangerous(k))
            || (s < self.hold_until * p && self.nb[sq][..4].iter().any(|&k| is_dangerous(k)))
        {
            let mut best: Option<((f64, f64), usize)> = None;
            for d in 0..5 {
                let m = self.nb[sq][d];
                let danger: i64 = if self.dest[m] == 0 {
                    self.nb[m]
                        .iter()
                        .map(|&k| {
                            dangerous
                                .iter()
                                .find(|&&(dk, _)| dk == k)
                                .map_or(0, |&(_, v)| v)
                        })
                        .sum()
                } else {
                    0
                };
                let v = self.pfv(m)
                    + (10000 * (self.dest[m] + s - 255).max(0)) as f64
                    + self.cfg.danger * danger as f64
                    + if self.wall[m] { 1e7 } else { 0.0 }
                    + if self.owner[m] == 0 && s <= self.strn[m] {
                        100000.0
                    } else {
                        0.0
                    };
                let r = self.rng.random();
                if best.map_or(true, |(b, _)| tuple_cmp((v, r), b) == Ordering::Less) {
                    best = Some(((v, r), d));
                }
            }
            return best.unwrap().1;
        }

        if stay_loss > 0 || potential > 9000.0 {
            let dt = self.dest[target];
            if (s + p + self.dest[sq]).min(255) + dt.min(255)
                > self.dest[sq].min(255) + (dt + s).min(255)
            {
                return PSTILL; // losing strength by staying beats losing more by moving
            }
            return best_d;
        }

        if !self.mine(target) {
            if (s == 255 || s + self.dest[target] > self.strn[target])
                && s >= 2 * p
                && (self.dest[target] == 0 || s >= self.hold_until * p)
            {
                return best_d;
            }
        } else if s >= self.strength_hurdle.max(self.hold_until * p) {
            return best_d;
        }
        PSTILL
    }

    fn turn(&mut self, map: &GameMap) -> Vec<Move> {
        let n = self.n;
        for i in 0..n {
            let s = map.contents[self.ys[i]][self.xs[i]];
            self.owner[i] = s.owner;
            self.strn[i] = s.strength as i64;
        }
        if self.owner.iter().all(|&o| o == self.me) {
            return Vec::new();
        }
        let mut rivals = [false; 256];
        for &o in &self.owner {
            if o != 0 && o != self.me {
                rivals[o as usize] = true;
            }
        }
        self.duel = rivals.iter().filter(|&&present| present).count() <= 1;
        self.degradation = if self.cfg.auto && self.duel {
            0.2
        } else {
            self.cfg.degradation
        };
        self.production_power = if self.cfg.auto && self.duel {
            1.0
        } else {
            self.cfg.prod_power
        };
        let touches_me = |b: &Bot, i: usize| b.nb[i][..4].iter().any(|&m| b.mine(m));
        let hero = (0..n).any(|i| self.is_empty(i) && touches_me(self, i));
        self.hold_until = if hero {
            if self.cfg.auto && self.duel {
                self.cfg.duel_hold
            } else {
                self.cfg.combat_hold
            }
        } else {
            self.cfg.mining_hold
        };
        for i in 0..n {
            self.wall[i] = self.cfg.walls
                && !(self.cfg.auto && self.duel)
                && self.owner[i] == 0
                && self.strn[i] > 0
                && touches_me(self, i)
                && self.nb[i][..4]
                    .iter()
                    .any(|&m| self.is_enemy(m) || self.is_empty(m));
        }
        // Break a diplomatic wall only when our economy and reserves support it.
        // Each adjacent rival is assessed separately, avoiding an all-fronts war.
        if self.cfg.breach > 0.0 {
            let mut power = [0.0f64; 256];
            for i in 0..n {
                if self.owner[i] != 0 {
                    power[self.owner[i] as usize] +=
                        self.strn[i] as f64 + 10.0 * self.prod[i] as f64;
                }
            }
            for i in 0..n {
                if !self.wall[i] {
                    continue;
                }
                let rivals: Vec<u8> = self.nb2[i]
                    .iter()
                    .filter(|&&j| self.is_enemy(j))
                    .map(|&j| self.owner[j])
                    .collect();
                if !rivals.is_empty()
                    && rivals
                        .iter()
                        .all(|&r| power[self.me as usize] > self.cfg.breach * power[r as usize])
                {
                    self.wall[i] = false;
                }
            }
        }
        self.mining_remains = (0..n).any(|i| {
            self.owner[i] == 0 && self.prod[i] > 0 && !self.wall[i] && touches_me(self, i)
        });

        self.potential_field();
        self.red_green();

        let mut interior: Vec<i64> = (0..n)
            .filter(|&i| self.mine(i) && self.mine(self.best_neighbor(i)))
            .map(|i| self.strn[i])
            .collect();
        interior.sort_by(|a, b| b.cmp(a));
        let int_max = if self.cfg.auto && self.duel {
            self.cfg.duel_int_max
        } else {
            self.cfg.int_max
        };
        let percentile = (1.0 - interior.len() as f64 / (50 * 50) as f64)
            * (int_max - self.cfg.int_min)
            + self.cfg.int_min;
        self.strength_hurdle = if interior.is_empty() {
            0
        } else {
            interior[((interior.len() as f64 * percentile.clamp(0.0, 0.999)) as usize)
                .min(interior.len() - 1)]
        };

        let mut order: Vec<usize> = (0..n)
            .filter(|&i| self.mine(i) && self.strn[i] > 0)
            .collect();
        // sorted(key=(strength, -distance), reverse=True), stable
        order.sort_by(|&a, &b| (self.strn[b], -self.pf[b].1).cmp(&(self.strn[a], -self.pf[a].1)));
        for v in self.dest.iter_mut() {
            *v = 0;
        }
        let mut moves = Vec::with_capacity(order.len());
        for sq in order {
            let d = self.assign_move(sq);
            let t = self.nb[sq][d];
            self.dest[t] += self.strn[sq] + if d == PSTILL { self.prod[sq] } else { 0 };
            moves.push(Move {
                loc: Location {
                    x: self.xs[sq] as u16,
                    y: self.ys[sq] as u16,
                },
                direction: ((d + 1) % 5) as u8,
            });
        }
        moves
    }
}

fn make_bot(me: u8, init_map: &GameMap) -> Bot {
    let (w, h) = (init_map.width as usize, init_map.height as usize);
    let n = w * h;
    let mut xs = vec![0; n];
    let mut ys = vec![0; n];
    let mut prod = vec![0i64; n];
    let mut nb = vec![[0usize; 5]; n];
    let mut nb2 = vec![Vec::new(); n];
    let mut seed: u64 = 0xcbf2_9ce4_8422_2325 ^ me as u64;
    let at =
        |x: i64, y: i64| (y.rem_euclid(h as i64) as usize) * w + x.rem_euclid(w as i64) as usize;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            xs[i] = x;
            ys[i] = y;
            let s = init_map.contents[y][x];
            prod[i] = s.production as i64;
            seed = (seed ^ (s.production as u64 * 256 + s.strength as u64))
                .wrapping_mul(0x100_0000_01b3);
            let (xi, yi) = (x as i64, y as i64);
            nb[i] = [
                at(xi, yi - 1),
                at(xi + 1, yi),
                at(xi, yi + 1),
                at(xi - 1, yi),
                i,
            ];
            for dy in -2i64..=2 {
                for dx in -2i64..=2 {
                    if dx.abs() + dy.abs() <= 2 && (dx != 0 || dy != 0) {
                        nb2[i].push(at(xi + dx, yi + dy));
                    }
                }
            }
        }
    }
    let rng = match std::env::var("ERDMAN_SEED")
        .ok()
        .and_then(|v| v.trim().parse::<i128>().ok())
    {
        Some(s) => PyRandom::from_int(s),
        None => PyRandom::from_int((seed >> 1) as i128),
    };
    Bot {
        cfg: config(),
        duel: false,
        production_power: config().prod_power,
        degradation: config().degradation,
        n,
        me,
        xs,
        ys,
        nb,
        nb2,
        prod,
        rng,
        owner: vec![0; n],
        strn: vec![0; n],
        wall: vec![false; n],
        mining_remains: false,
        pf: vec![(0.0, 0); n],
        dest: vec![0; n],
        redlight: vec![false; n],
        greenlight: vec![false; n],
        hold_until: 5,
        strength_hurdle: 0,
    }
}

fn main() {
    let (me, init_map, mut conn) = get_init();
    let mut bot = make_bot(me, &init_map);
    send_init("CodexBot");
    loop {
        let map = conn.get_frame();
        let moves = bot.turn(&map);
        conn.send_frame(&moves);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplayer_becomes_a_duel_when_one_rival_is_eliminated() {
        let mut map = GameMap {
            width: 7,
            height: 7,
            contents: vec![
                vec![
                    Site {
                        owner: 0,
                        production: 4,
                        strength: 20
                    };
                    7
                ];
                7
            ],
        };
        map.contents[2][2].owner = 1;
        map.contents[2][2].strength = 255;
        map.contents[2][4].owner = 2;
        map.contents[6][6].owner = 3;
        let mut bot = make_bot(1, &map);
        bot.turn(&map);
        assert!(!bot.duel);
        assert!(bot.wall[2 * 7 + 3]);
        assert_eq!(bot.degradation, 0.4);
        map.contents[6][6].owner = 0;
        bot.turn(&map);
        assert!(bot.duel);
        assert!(!bot.wall[2 * 7 + 3]);
        assert_eq!(bot.production_power, 1.0);
        assert_eq!(bot.degradation, 0.2);
    }

    #[test]
    fn fully_owned_map_has_no_frontier_and_does_not_panic() {
        let map = GameMap {
            width: 5,
            height: 7,
            contents: vec![
                vec![
                    Site {
                        owner: 1,
                        production: 5,
                        strength: 200
                    };
                    5
                ];
                7
            ],
        };
        let mut bot = make_bot(1, &map);
        assert!(bot.turn(&map).is_empty());
    }

    #[test]
    fn zero_production_and_empty_frontier_produce_valid_moves() {
        let mut map = GameMap {
            width: 5,
            height: 7,
            contents: vec![
                vec![
                    Site {
                        owner: 0,
                        production: 0,
                        strength: 0
                    };
                    5
                ];
                7
            ],
        };
        map.contents[0][0] = Site {
            owner: 1,
            production: 0,
            strength: 255,
        };
        map.contents[6][4] = Site {
            owner: 2,
            production: 0,
            strength: 255,
        };
        let mut bot = make_bot(1, &map);
        for m in bot.turn(&map) {
            assert!(m.direction <= WEST);
            assert_eq!(map.contents[m.loc.y as usize][m.loc.x as usize].owner, 1);
        }
    }
}
