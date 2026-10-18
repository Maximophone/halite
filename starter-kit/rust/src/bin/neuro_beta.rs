//! NeuroBeta: NeuroAlpha's network, plus ApexBeta's one-turn combat search at the front.
//!
//!   1. The network (src/nn.rs) gives every move: ours, and each nearby enemy's (the network run
//!      from that enemy's seat: "the enemy plays like us").
//!   2. Our pieces within NEURO_ZONE steps of an enemy are refined by an exact one-turn
//!      simulation (moves, merges, the 255 cap, overkill, neutral damage, who holds each site),
//!      scored as enemy strength destroyed - ours lost + production of the sites held, minus the
//!      enemy's, over two scenarios mixed by NEURO_PRED: the predicted enemy moves, and every
//!      enemy staying still. Coordinate ascent, strongest piece first, NEURO_PASSES passes;
//!      leaving the network's move needs a margin of NEURO_BIAS.
//! The search and its scoring are ApexBeta's (apex_beta.rs) with its tuned defaults. It is also
//! the improvement operator of the expert iteration in rl/: the network learns the search's moves.
//! NEURO_SEARCH=0 plays exactly as NeuroAlpha. NEURO_WEIGHTS=path loads other weights.

use hlt::nn::{Net, NUM_MOVES};
use hlt::*;

const WEIGHTS: &[u8] = include_bytes!("../../weights/neuro_alpha.hnn");

struct Params {
    search: bool,
    zone: i32,
    pred: f64,
    lambda: f64,
    lambda_terr: f64,
    terr: f64,
    neutral_w: f64,
    bias: f64,
    passes: usize,
}

fn env(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

impl Params {
    fn load() -> Params {
        Params {
            search: env("NEURO_SEARCH", 1.0) > 0.0,
            zone: env("NEURO_ZONE", 3.0) as i32,
            pred: env("NEURO_PRED", 0.6),
            lambda: env("NEURO_LAMBDA", 1.0),
            lambda_terr: env("NEURO_LAMBDA_TERR", 1.0),
            terr: env("NEURO_TERR", 2.0),
            neutral_w: env("NEURO_NEUTRAL_W", 0.3),
            bias: env("NEURO_BIAS", 1.0),
            passes: env("NEURO_PASSES", 3.0) as usize,
        }
    }
}

struct Bot {
    w: usize,
    h: usize,
    n: usize,
    net: Net,
    p: Params,
    /// neighbours in NORTH, EAST, SOUTH, WEST order (index = direction - 1)
    nb: Vec<[usize; 4]>,
    prod: Vec<i32>,
    prod_u8: Vec<u8>,
}

/// One-turn combat simulation around the front (ApexBeta's `Sim`).
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
    /// The network's most likely move for every site of player `me` (STILL elsewhere).
    fn policy(&self, me: i32, owner: &[i32], strength: &[i32], turn: usize) -> Vec<u8> {
        let own: Vec<u8> = owner.iter().map(|&o| o as u8).collect();
        let st: Vec<u8> = strength.iter().map(|&s| s as u8).collect();
        let logits = self.net.forward(self.w, self.h, &self.net.features(self.w, self.h, &own, &st, &self.prod_u8, me as u8, turn));
        let mut mv = vec![STILL; self.n];
        for i in 0..self.n {
            if owner[i] == me {
                let l = &logits[i * NUM_MOVES..(i + 1) * NUM_MOVES];
                mv[i] = (0..NUM_MOVES).max_by(|&a, &b| l[a].total_cmp(&l[b]).then(b.cmp(&a))).unwrap() as u8;
            }
        }
        mv
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

    /// ApexBeta's combat search over our pieces near an enemy, starting from the network's moves.
    fn combat(&self, me: i32, owner: &[i32], strength: &[i32], mv: &mut [u8], turn: usize) {
        let (n, p) = (self.n, &self.p);
        let zone = p.zone;
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
        // Scenario 0: every enemy stays. Scenario 1: each rival plays the network.
        let mut pred = vec![STILL; n];
        if p.pred > 0.0 {
            for &q in &rivals {
                let m = self.policy(q, owner, strength, turn);
                for i in 0..n {
                    if owner[i] == q {
                        pred[i] = m[i];
                    }
                }
            }
        }
        let mut e_s = vec![vec![-1i32; n]; 2];
        let mut e_hit = vec![vec![0i32; n]; 2];
        for k in 0..2 {
            let mut raw = vec![0i32; n];
            let mut here = vec![false; n];
            for i in 0..n {
                if owner[i] == 0 || owner[i] == me {
                    continue;
                }
                let d = if k == 0 { STILL } else { pred[i] };
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
        let mut sim = Sim { bot: self, me, owner, strength, o_raw: vec![0; n], arr: vec![0; n], e_s, e_hit, weight: vec![1.0 - p.pred, p.pred] };
        for i in 0..n {
            if owner[i] == me {
                sim.apply(i, mv[i], 1);
            }
        }
        let default: Vec<u8> = pieces.iter().map(|&i| mv[i]).collect();
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
                    let sites = [i, sim.dest(i, cur), sim.dest(i, d)];
                    let before = sim.local(&sites, &mut buf);
                    sim.apply(i, cur, -1);
                    sim.apply(i, d, 1);
                    let after = sim.local(&sites, &mut buf);
                    sim.apply(i, d, -1);
                    sim.apply(i, cur, 1);
                    // Leaving the network's move needs a margin: it also serves expansion and
                    // routing, which this score ignores.
                    let bias = if d == default[pi] { 0.0 } else if cur == default[pi] { p.bias } else { 0.0 };
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
    }
}

fn main() {
    let (me, map, mut conn) = get_init();
    let (w, h) = (map.width as usize, map.height as usize);
    let n = w * h;
    let bytes = match std::env::var("NEURO_WEIGHTS") {
        Ok(p) => std::fs::read(&p).unwrap_or_else(|e| panic!("cannot read {p}: {e}")),
        Err(_) => WEIGHTS.to_vec(),
    };
    let net = Net::load(&bytes).unwrap_or_else(|e| panic!("bad weights: {e}"));
    let mut nb = vec![[0usize; 4]; n];
    let mut prod = vec![0i32; n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            nb[i] = [((y + h - 1) % h) * w + x, y * w + (x + 1) % w, ((y + 1) % h) * w + x, y * w + (x + w - 1) % w];
            prod[i] = map.contents[y][x].production as i32;
        }
    }
    let prod_u8 = prod.iter().map(|&p| p as u8).collect();
    let bot = Bot { w, h, n, net, p: Params::load(), nb, prod, prod_u8 };
    send_init("NeuroBeta");

    let me = me as i32;
    let mut turn = 0;
    loop {
        let map = conn.get_frame();
        let mut owner = vec![0i32; n];
        let mut strength = vec![0i32; n];
        for y in 0..h {
            for x in 0..w {
                let s = map.contents[y][x];
                owner[y * w + x] = s.owner as i32;
                strength[y * w + x] = s.strength as i32;
            }
        }
        let mut mv = bot.policy(me, &owner, &strength, turn);
        if bot.p.search {
            bot.combat(me, &owner, &strength, &mut mv, turn);
        }
        let moves: Vec<Move> = (0..n)
            .filter(|&i| owner[i] == me && mv[i] != STILL)
            .map(|i| Move { loc: Location { x: (i % w) as u16, y: (i / w) as u16 }, direction: mv[i] })
            .collect();
        conn.send_frame(&moves);
        turn += 1;
    }
}
