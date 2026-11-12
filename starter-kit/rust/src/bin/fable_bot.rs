//! FableBot: CodexBot's policy (a tuned ErdmanBot, see `codex_bot.rs` and `erdman_bot.rs`;
//! Travis Erdman's 2016 winning strategy, https://github.com/erdman/erdman-halite-bots) plus:
//!
//! - Barren land: ErdmanBot treats zero-production squares as impassable, so a bot that starts
//!   among them never leaves. Here they count as production `FABLE_ZERO_PROD` (0.25).
//! - Duels: a one-turn combat search (ApexBeta's, as in `neuro_search.rs`) over our pieces
//!   within 3 steps of an enemy. Enemy moves are predicted by running this same policy from the
//!   enemy's seat; no opponent is ever identified.
//! - Multiplayer: the search is off while the walls stand (it starts fights that the
//!   non-aggression walls prevent). After `FABLE_WAR_IDLE` turns without a new territory
//!   maximum, the bot goes to war with its weakest neighbour if it is at least as strong as it
//!   and as its other neighbours: walls facing that rival are released and the search runs
//!   against that rival only.
//! - Multiplayer constants: routing distance penalty 0.3, no strategic stilling.
//!
//! Every constant reads a `FABLE_*` variable; `FABLE_DEBUG=1` logs each turn to stderr.
//! Measurements are in CLAUDE.md ("FableBot").

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
    /// Production a zero-production square counts as in the potential field (0: impassable,
    /// as in ErdmanBot, which leaves a bot that starts in barren land stuck for the game).
    zero_prod: f64,
}
fn config() -> &'static Config {
    static CONFIG: std::sync::OnceLock<Config> = std::sync::OnceLock::new();
    CONFIG.get_or_init(|| {
        let get = |key: &str, default: f64| {
            std::env::var(format!("FABLE_{key}"))
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| v.is_finite())
                .unwrap_or(default)
        };
        Config {
            alpha: get("ALPHA", 0.1),
            degradation: get("DEGRADE", 0.3),
            enemy_roi: get("ENEMY", -0.5),
            combat_hold: get("HOLD", 7.0) as i64,
            int_max: get("INT_MAX", 0.45),
            int_min: get("INT_MIN", 0.01),
            mining_hold: get("MINE_HOLD", 5.0) as i64,
            danger: get("DANGER", 5000.0),
            strategic: get("STRATEGIC", 0.0) != 0.0,
            walls: get("WALLS", 1.0) != 0.0,
            prod_power: get("PROD_POWER", 1.0),
            breach: get("BREACH", 0.0),
            auto: get("AUTO", 1.0) != 0.0,
            duel_hold: get("DUEL_HOLD", 8.0) as i64,
            duel_int_max: get("DUEL_INT_MAX", 0.75),
            zero_prod: get("ZERO_PROD", 0.25),
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
    /// Multiplayer war: the rivals we attack (walls facing them are released).
    hostile: [bool; 256],
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

    /// Production as the potential field sees it.
    fn pe(&self, i: usize) -> f64 {
        if self.prod[i] == 0 {
            self.cfg.zero_prod
        } else {
            self.prod[i] as f64
        }
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
        } else if self.pe(i) == 0.0 || self.is_enemy(i) {
            f64::INFINITY
        } else if self.wall[i] {
            (100 * self.strn[i]) as f64 / self.pe(i).powf(self.production_power)
        } else {
            self.strn[i] as f64 / self.pe(i).powf(self.production_power)
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
                    let np = if self.pe(m) != 0.0 && self.owner[m] == 0 {
                        (1.0 - self.cfg.alpha) * e.pot
                            + self.cfg.alpha * self.strn[m] as f64
                                / self.pe(m).powf(self.production_power)
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

    /// Moves of seat `me` (hlt directions, STILL where it has no piece or stays).
    fn policy(&mut self, me: u8, owner: &[u8], strn: &[i64]) -> Vec<u8> {
        let n = self.n;
        self.me = me;
        self.owner.copy_from_slice(owner);
        self.strn.copy_from_slice(strn);
        let mut mv = vec![STILL; n];
        if self.owner.iter().all(|&o| o == self.me) {
            return mv;
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
        if self.hostile.iter().any(|&h| h) {
            for i in 0..n {
                if self.wall[i]
                    && self.nb2[i]
                        .iter()
                        .any(|&j| self.is_enemy(j) && self.hostile[self.owner[j] as usize])
                {
                    self.wall[i] = false;
                }
            }
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
            self.owner[i] == 0 && self.pe(i) > 0.0 && !self.wall[i] && touches_me(self, i)
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
        for sq in order {
            let d = self.assign_move(sq);
            let t = self.nb[sq][d];
            self.dest[t] += self.strn[sq] + if d == PSTILL { self.prod[sq] } else { 0 };
            mv[sq] = ((d + 1) % 5) as u8;
        }
        mv
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
        hostile: [false; 256],
    }
}

// ---------------------------------------------------------------------------------------------
// One-turn combat search (ApexBeta's, as in neuro_search.rs) over our pieces near an enemy.

struct Params {
    search: bool,
    zone: i64,
    pred: f64,
    lambda: f64,
    lambda_terr: f64,
    terr: f64,
    neutral_w: f64,
    bias: f64,
    passes: usize,
    ffa_search: bool,
    /// In multiplayer, the search may not enter a neutral site that still has strength
    /// (1), nor any site the policy did not choose that is not ours (2).
    ffa_safe: i64,
    /// War in multiplayer: attack the weakest neighbour once we have `war_ratio` times its
    /// strength and nothing was left to mine for `war_idle` turns (0: never).
    war_ratio: f64,
    war_idle: i64,
    /// Attack even while mining remains with this strength ratio (0: never).
    war_early: f64,
    /// War only if we have this share of the strength of our strongest other neighbour.
    war_safe: f64,
}

fn env(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

impl Params {
    fn load() -> Params {
        Params {
            search: env("FABLE_SEARCH", 1.0) > 0.0,
            zone: env("FABLE_ZONE", 3.0) as i64,
            pred: env("FABLE_PRED", 0.8),
            lambda: env("FABLE_LAMBDA", 1.0),
            lambda_terr: env("FABLE_LAMBDA_TERR", 1.0),
            terr: env("FABLE_TERR", 2.0),
            neutral_w: env("FABLE_NEUTRAL_W", 0.3),
            bias: env("FABLE_BIAS", 1.0),
            passes: env("FABLE_PASSES", 3.0) as usize,
            ffa_search: env("FABLE_FFA_SEARCH", 0.0) > 0.0,
            ffa_safe: env("FABLE_FFA_SAFE", 1.0) as i64,
            war_ratio: env("FABLE_WAR_RATIO", 1.0),
            war_idle: env("FABLE_WAR_IDLE", 15.0) as i64,
            war_early: env("FABLE_WAR_EARLY", 0.0),
            war_safe: env("FABLE_WAR_SAFE", 1.0),
        }
    }
}

struct Sim<'a> {
    p: &'a Params,
    nb: &'a [[usize; 5]],
    prod: &'a [i64],
    me: u8,
    owner: &'a [u8],
    strength: &'a [i64],
    /// Our strength arriving on (or staying at) each site, before the cap.
    o_raw: Vec<i64>,
    /// Our pieces moving onto each site (a site we own always holds a piece).
    arr: Vec<i64>,
    /// Per scenario: enemy strength on each site after their moves (-1: none),
    /// and the enemy strength that hits a piece standing there.
    e_s: Vec<Vec<i64>>,
    e_hit: Vec<Vec<i64>>,
    weight: Vec<f64>,
}

impl<'a> Sim<'a> {
    fn dest(&self, i: usize, d: u8) -> usize {
        if d == STILL {
            i
        } else {
            self.nb[i][d as usize - 1]
        }
    }
    fn ours(&self, c: usize) -> i64 {
        if self.owner[c] == self.me || self.arr[c] > 0 {
            self.o_raw[c].min(255)
        } else {
            -1
        }
    }
    fn apply(&mut self, i: usize, d: u8, sign: i64) {
        let s = self.strength[i] + if d == STILL { self.prod[i] } else { 0 };
        let t = self.dest(i, d);
        self.o_raw[t] += sign * s;
        if d != STILL {
            self.arr[t] += sign;
        }
    }
    fn cell(&self, c: usize, k: usize) -> f64 {
        let p = self.p;
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
            for &j in &self.nb[c] {
                de += self.ours(j).max(0);
            }
            score += p.lambda * es.min(de) as f64;
            en_alive = es - de > 0 || (es == 0 && de == 0);
        }
        let v = p.terr * (self.prod[c] as f64 + 0.5);
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
            for &j in &self.nb[x] {
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

fn bfs(nb: &[[usize; 5]], seeds: impl Iterator<Item = usize>) -> Vec<i64> {
    let mut dist = vec![i64::MAX; nb.len()];
    let mut layer: Vec<usize> = seeds.collect();
    for &i in &layer {
        dist[i] = 0;
    }
    let mut d = 0;
    while !layer.is_empty() {
        d += 1;
        let mut next = Vec::new();
        for &i in &layer {
            for &j in &nb[i][..4] {
                if dist[j] == i64::MAX {
                    dist[j] = d;
                    next.push(j);
                }
            }
        }
        layer = next;
    }
    dist
}

/// Refines `mv` (our policy moves) for our pieces within `zone` of an enemy. Returns how many
/// moves changed.
fn combat(
    bot: &mut Bot,
    p: &Params,
    me: u8,
    owner: &[u8],
    strength: &[i64],
    mv: &mut [u8],
    hostile: Option<&[bool; 256]>,
    walls: &[bool],
) -> usize {
    let n = bot.n;
    let zone = p.zone;
    let foe = |o: u8| o != 0 && o != me && hostile.map_or(true, |h| h[o as usize]);
    let edist = bfs(&bot.nb, (0..n).filter(|&i| foe(owner[i])));
    let mut pieces: Vec<usize> = (0..n)
        .filter(|&i| owner[i] == me && strength[i] > 0 && edist[i] <= zone)
        .collect();
    if pieces.is_empty() {
        return 0;
    }
    pieces.sort_by_key(|&i| (-strength[i], i));
    let mdist = bfs(&bot.nb, (0..n).filter(|&i| owner[i] == me));
    let mut rivals: Vec<u8> = Vec::new();
    for i in 0..n {
        if owner[i] != 0 && owner[i] != me && mdist[i] <= zone && !rivals.contains(&owner[i]) {
            rivals.push(owner[i]);
        }
    }
    rivals.sort();
    // Scenario 0: every enemy stays. Scenario 1: each rival plays our own policy from its seat.
    let mut pred = vec![STILL; n];
    if p.pred > 0.0 {
        for &q in &rivals {
            let saved = bot.hostile;
            bot.hostile = [false; 256];
            let m = bot.policy(q, owner, strength);
            bot.hostile = saved;
            for i in 0..n {
                if owner[i] == q {
                    pred[i] = m[i];
                }
            }
        }
    }
    let mut e_s = vec![vec![-1i64; n]; 2];
    let mut e_hit = vec![vec![0i64; n]; 2];
    for k in 0..2 {
        let mut raw = vec![0i64; n];
        let mut here = vec![false; n];
        for i in 0..n {
            if owner[i] == 0 || owner[i] == me {
                continue;
            }
            let d = if k == 0 { STILL } else { pred[i] };
            here[i] = true;
            if d == STILL {
                raw[i] += strength[i] + bot.prod[i];
            } else {
                let t = bot.nb[i][d as usize - 1];
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
                for &j in &bot.nb[i] {
                    e_hit[k][j] += s;
                }
            }
        }
    }
    let mut sim = Sim {
        p,
        nb: &bot.nb,
        prod: &bot.prod,
        me,
        owner,
        strength,
        o_raw: vec![0; n],
        arr: vec![0; n],
        e_s,
        e_hit,
        weight: vec![1.0 - p.pred, p.pred],
    };
    for i in 0..n {
        if owner[i] == me {
            sim.apply(i, mv[i], 1);
        }
    }
    let default: Vec<u8> = pieces.iter().map(|&i| mv[i]).collect();
    let safe = if bot.duel { 0 } else { p.ffa_safe };
    let mut buf = Vec::new();
    for _ in 0..p.passes {
        let mut changed = false;
        for (pi, &i) in pieces.iter().enumerate() {
            let cur = mv[i];
            let mut best = cur;
            let mut best_gain = 0.0;
            for d in DIRECTIONS {
                if d == cur {
                    continue;
                }
                if safe > 0 && d != default[pi] && d != STILL {
                    let t = sim.dest(i, d);
                    if (owner[t] == 0 && strength[t] > 0 && (hostile.is_none() || walls[t]))
                        || (safe > 1 && owner[t] != me)
                    {
                        continue;
                    }
                }
                let sites = [i, sim.dest(i, cur), sim.dest(i, d)];
                let before = sim.local(&sites, &mut buf);
                sim.apply(i, cur, -1);
                sim.apply(i, d, 1);
                let after = sim.local(&sites, &mut buf);
                sim.apply(i, d, -1);
                sim.apply(i, cur, 1);
                // Leaving the policy's move needs a margin: it also serves expansion and
                // routing, which this score ignores.
                let bias = if d == default[pi] {
                    0.0
                } else if cur == default[pi] {
                    p.bias
                } else {
                    0.0
                };
                let gain = after - before - bias;
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
        if !changed {
            break;
        }
    }
    pieces
        .iter()
        .zip(&default)
        .filter(|&(&i, &d)| mv[i] != d)
        .count()
}

fn main() {
    let (me, init_map, mut conn) = get_init();
    let mut bot = make_bot(me, &init_map);
    let params = Params::load();
    let debug = std::env::var("FABLE_DEBUG").is_ok();
    send_init("FableBot");
    let n = bot.n;
    let mut owner = vec![0u8; n];
    let mut strn = vec![0i64; n];
    let mut turn = 0;
    let mut idle = 0i64;
    let mut best_terr = 0usize;
    let mut target: Option<u8> = None;
    loop {
        let map = conn.get_frame();
        let t0 = std::time::Instant::now();
        for i in 0..n {
            let s = map.contents[bot.ys[i]][bot.xs[i]];
            owner[i] = s.owner;
            strn[i] = s.strength as i64;
        }
        // War and peace in multiplayer.
        let mut power = [0i64; 256];
        for i in 0..n {
            power[owner[i] as usize] += strn[i];
        }
        let rivals = (1..256).filter(|&q| q != me as usize && power[q] > 0).count();
        if let Some(q) = target {
            if power[q as usize] == 0 || rivals <= 1 {
                target = None;
            }
        }
        if target.is_none() && rivals > 1 && params.war_idle > 0 {
            let ratio = if idle >= params.war_idle {
                params.war_ratio
            } else {
                params.war_early
            };
            if ratio > 0.0 {
                let mdist = bfs(&bot.nb, (0..n).filter(|&i| owner[i] == me));
                let mut near = [false; 256];
                for i in 0..n {
                    if owner[i] != 0 && owner[i] != me && mdist[i] <= 3 {
                        near[owner[i] as usize] = true;
                    }
                }
                let best = (1..256usize).filter(|&q| near[q]).min_by_key(|&q| power[q]);
                if let Some(q) = best {
                    // Not while a neighbour much stronger than us would pick up the pieces.
                    let others = (1..256usize)
                        .filter(|&r| near[r] && r != q)
                        .map(|r| power[r])
                        .max()
                        .unwrap_or(0);
                    if power[me as usize] as f64 >= ratio * power[q] as f64
                        && power[me as usize] as f64 >= params.war_safe * others as f64
                    {
                        target = Some(q as u8);
                    }
                }
            }
        }
        bot.hostile = [false; 256];
        if let Some(q) = target {
            bot.hostile[q as usize] = true;
        }
        let mut mv = bot.policy(me, &owner, &strn);
        // Idle: turns since our territory last reached a new maximum.
        let terr = owner.iter().filter(|&&o| o == me).count();
        if terr > best_terr {
            best_terr = terr;
            idle = 0;
        } else {
            idle += 1;
        }
        let duel = bot.duel;
        let walls = bot.wall.clone();
        let hostile = bot.hostile;
        let mut changed = 0;
        if params.search && (duel || params.ffa_search) {
            changed = combat(&mut bot, &params, me, &owner, &strn, &mut mv, None, &walls);
        } else if params.search && target.is_some() {
            changed = combat(&mut bot, &params, me, &owner, &strn, &mut mv, Some(&hostile), &walls);
        }
        if debug {
            eprintln!(
                "turn {turn} changed {changed} idle {idle} target {target:?} ms {:.1}",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        let moves: Vec<Move> = (0..n)
            .filter(|&i| owner[i] == me && mv[i] != STILL)
            .map(|i| Move {
                loc: Location {
                    x: bot.xs[i] as u16,
                    y: bot.ys[i] as u16,
                },
                direction: mv[i],
            })
            .collect();
        conn.send_frame(&moves);
        turn += 1;
    }
}
