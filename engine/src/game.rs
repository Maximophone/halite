//! Map generator and rules: a line-by-line port of `generate_map` and
//! `GameState` in halite.py. Grids are flat, row-major (index y * W + x).

use crate::pyrandom::PyRandom;

pub const STILL: u8 = 0;
pub const NORTH: u8 = 1;
pub const EAST: u8 = 2;
pub const SOUTH: u8 = 3;
pub const WEST: u8 = 4;
pub const MAX_STRENGTH: i32 = 255;
pub const START_STRENGTH: i32 = 255;
pub const MAX_PLAYERS: usize = 6;

// ---------------------------------------------------------------------------
// Map generation
// ---------------------------------------------------------------------------

/// Multi-octave value noise in [0, 1]; same float operations, same order.
fn smooth_noise(w: usize, h: usize, rng: &mut PyRandom) -> Vec<Vec<f64>> {
    let (octaves, start_cells) = (4u32, 2usize);
    let mut out = vec![vec![0.0f64; w]; h];
    let mut amp = 1.0f64;
    for octave in 0..octaves {
        let g = start_cells * (1 << octave);
        let mut grid = Vec::with_capacity(g + 1);
        for _ in 0..g + 1 {
            let row: Vec<f64> = (0..g + 1).map(|_| rng.random()).collect();
            grid.push(row);
        }
        let gf = g as f64;
        for y in 0..h {
            let fy = y as f64 / h as f64 * gf;
            let iy = fy as usize;
            let mut ty = fy - iy as f64;
            ty = ty * ty * (3.0 - 2.0 * ty);
            let (row_a, row_b) = (&grid[iy], &grid[iy + 1]);
            let out_row = &mut out[y];
            for x in 0..w {
                let fx = x as f64 / w as f64 * gf;
                let ix = fx as usize;
                let mut tx = fx - ix as f64;
                tx = tx * tx * (3.0 - 2.0 * tx);
                let top = row_a[ix] + (row_a[ix + 1] - row_a[ix]) * tx;
                let bot = row_b[ix] + (row_b[ix + 1] - row_b[ix]) * tx;
                out_row[x] += amp * (top + (bot - top) * ty);
            }
        }
        amp *= 0.5;
    }
    // Python's min()/max() keep the first of equal values; for floats that
    // only matters for -0.0 versus 0.0, which cannot occur here.
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for row in &out {
        let rlo = row.iter().cloned().fold(f64::INFINITY, |a, b| if b < a { b } else { a });
        let rhi = row.iter().cloned().fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
        if rlo < lo {
            lo = rlo;
        }
        if rhi > hi {
            hi = rhi;
        }
    }
    let mut span = hi - lo;
    if span == 0.0 {
        span = 1.0;
    }
    out.iter()
        .map(|row| row.iter().map(|v| (v - lo) / span).collect())
        .collect()
}

pub fn layout(num_players: usize) -> (i64, i64) {
    match num_players {
        1 => (1, 1),
        2 => (1, 2),
        3 => (1, 3),
        4 => (2, 2),
        5 => (1, 5),
        _ => (2, 3),
    }
}

pub struct Map {
    pub width: usize,
    pub height: usize,
    pub production: Vec<i32>,
    pub strength: Vec<i32>,
    pub starts: Vec<(usize, usize)>,
}

/// Python's `int(round(x))`: round half to even.
fn py_round(x: f64) -> i32 {
    x.round_ties_even() as i32
}

pub fn generate_map(width: i64, height: i64, num_players: usize, rng: &mut PyRandom) -> Map {
    let (mut rows, mut cols) = layout(num_players);
    if width < height && cols > rows {
        std::mem::swap(&mut rows, &mut cols);
    }
    // Python's // is floor division.
    let tw = 3.max(width.div_euclid(cols)) as usize;
    let th = 3.max(height.div_euclid(rows)) as usize;
    let (rows, cols) = (rows as usize, cols as usize);
    let (w, h) = (tw * cols, th * rows);

    let n_prod = smooth_noise(tw, th, rng);
    let n_str = smooth_noise(tw, th, rng);

    let mut tile_prod = vec![vec![0i32; tw]; th];
    let mut tile_str = vec![vec![0i32; tw]; th];
    for y in 0..th {
        for x in 0..tw {
            let p = n_prod[y][x];
            tile_prod[y][x] = py_round(p.powf(2.2) * 15.0);
            let s = 0.45 * n_str[y][x] + 0.55 * p;
            let v = s.powf(2.0) * 280.0 + 5.0;
            // min(MAX_STRENGTH, v) returns the int 255 unless v < 255.
            let m = if v < MAX_STRENGTH as f64 { v } else { MAX_STRENGTH as f64 };
            tile_str[y][x] = py_round(m);
        }
    }

    let (sx, sy) = ((tw - 1) / 2, (th - 1) / 2);

    let mut production = vec![0i32; w * h];
    let mut strength = vec![0i32; w * h];
    let mut starts = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let (flip_x, flip_y) = (c % 2 == 1, r % 2 == 1);
            for y in 0..th {
                let ty = if flip_y { th - 1 - y } else { y };
                for x in 0..tw {
                    let tx = if flip_x { tw - 1 - x } else { x };
                    let i = (r * th + y) * w + c * tw + x;
                    production[i] = tile_prod[ty][tx];
                    strength[i] = tile_str[ty][tx];
                }
            }
            let px = c * tw + if flip_x { tw - 1 - sx } else { sx };
            let py = r * th + if flip_y { th - 1 - sy } else { sy };
            starts.push((px, py));
        }
    }
    rng.shuffle(&mut starts);
    starts.truncate(num_players);
    Map { width: w, height: h, production, strength, starts }
}

// ---------------------------------------------------------------------------
// Game state and rules
// ---------------------------------------------------------------------------

pub struct GameState {
    pub width: usize,
    pub height: usize,
    pub num_players: usize,
    pub production: Vec<i32>,
    pub strength: Vec<i32>,
    pub owner: Vec<u8>,
}

impl GameState {
    pub fn new(map: &Map) -> GameState {
        let mut g = GameState {
            width: map.width,
            height: map.height,
            num_players: map.starts.len(),
            production: map.production.clone(),
            strength: map.strength.clone(),
            owner: vec![0; map.width * map.height],
        };
        for (i, &(x, y)) in map.starts.iter().enumerate() {
            g.owner[y * g.width + x] = (i + 1) as u8;
            g.strength[y * g.width + x] = START_STRENGTH;
        }
        g
    }

    #[inline]
    fn neighbor(&self, i: usize, d: u8) -> usize {
        let (w, h) = (self.width, self.height);
        let (x, y) = (i % w, i / w);
        match d {
            NORTH => ((y + h - 1) % h) * w + x,
            SOUTH => ((y + 1) % h) * w + x,
            EAST => y * w + (x + 1) % w,
            WEST => y * w + (x + w - 1) % w,
            _ => i,
        }
    }

    /// Original Halite I map format: run-length owners, then strengths.
    pub fn serialize(&self) -> String {
        let mut out = String::with_capacity(self.owner.len() * 4 + 16);
        let mut count = 0usize;
        let mut current = self.owner[0];
        for &o in &self.owner {
            if o == current {
                count += 1;
            } else {
                push_int(&mut out, count as i64);
                out.push(' ');
                push_int(&mut out, current as i64);
                out.push(' ');
                count = 1;
                current = o;
            }
        }
        push_int(&mut out, count as i64);
        out.push(' ');
        push_int(&mut out, current as i64);
        for &s in &self.strength {
            out.push(' ');
            push_int(&mut out, s as i64);
        }
        out
    }

    /// (territory, strength, production) indexed by owner 0..=n.
    pub fn stats(&self) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        let n = self.num_players;
        let mut territory = vec![0i64; n + 1];
        let mut strength = vec![0i64; n + 1];
        let mut prod = vec![0i64; n + 1];
        for i in 0..self.owner.len() {
            let o = self.owner[i] as usize;
            territory[o] += 1;
            strength[o] += self.strength[i] as i64;
            prod[o] += self.production[i] as i64;
        }
        (territory, strength, prod)
    }

    /// Advance one turn. `dirs[i]` is the move of the piece on site i (only
    /// read for player-owned sites). See GameState.step in halite.py.
    pub fn step(&mut self, dirs: &[u8]) {
        let n = self.num_players;
        let size = self.owner.len();
        const NONE: i32 = -1;
        let owner = &self.owner;
        let strength = &self.strength;

        // pieces[p][i]: strength of player p's piece on site i, or NONE.
        let mut pieces = vec![vec![NONE; size]; n + 1];
        for i in 0..size {
            let o = owner[i] as usize;
            if o == 0 {
                continue;
            }
            let d = dirs[i];
            let pmap = &mut pieces[o];
            if d == STILL {
                let s = strength[i] + self.production[i];
                pmap[i] = MAX_STRENGTH.min(pmap[i].max(0) + s);
            } else {
                let dest = self.neighbor(i, d);
                pmap[dest] = MAX_STRENGTH.min(pmap[dest].max(0) + strength[i]);
                if pmap[i] == NONE {
                    pmap[i] = 0;
                }
            }
        }

        let mut damage = vec![vec![0i32; size]; n + 1];
        // neutral_damage[i] is NONE when no player piece entered neutral site i.
        let mut neutral_damage = vec![NONE; size];
        for p in 1..=n {
            for i in 0..size {
                let s = pieces[p][i];
                if s == NONE {
                    continue;
                }
                if s > 0 {
                    let targets = [
                        i,
                        self.neighbor(i, NORTH),
                        self.neighbor(i, EAST),
                        self.neighbor(i, SOUTH),
                        self.neighbor(i, WEST),
                    ];
                    for q in 1..=n {
                        if q == p {
                            continue;
                        }
                        for &t in &targets {
                            if pieces[q][t] != NONE {
                                damage[q][t] += s;
                            }
                        }
                    }
                }
                if owner[i] == 0 {
                    neutral_damage[i] = neutral_damage[i].max(0) + s;
                    let ns = strength[i];
                    if ns > 0 {
                        damage[p][i] += ns;
                    }
                }
            }
        }

        let mut new_owner = vec![0u8; size];
        let mut new_strength: Vec<i32> =
            (0..size).map(|i| if owner[i] == 0 { strength[i] } else { 0 }).collect();
        for i in 0..size {
            if neutral_damage[i] != NONE {
                let remaining = strength[i] - neutral_damage[i];
                new_strength[i] = if remaining > 0 { remaining } else { 0 };
            }
        }

        for i in 0..size {
            let mut count = 0;
            let mut sole = (0usize, 0i32);
            for p in 1..=n {
                let s = pieces[p][i];
                if s == NONE {
                    continue;
                }
                let dmg = damage[p][i];
                let remaining = s - dmg;
                if remaining > 0 || (dmg == 0 && s == 0) {
                    count += 1;
                    sole = (p, remaining.max(0));
                }
            }
            if count == 1 {
                if owner[i] == 0 && neutral_damage[i] != NONE && new_strength[i] > 0 {
                    continue; // neutral held the site
                }
                new_owner[i] = sole.0 as u8;
                new_strength[i] = sole.1;
            } else if count > 1 {
                new_owner[i] = 0;
                new_strength[i] = 0;
            }
        }

        self.owner = new_owner;
        self.strength = new_strength;
    }

    /// Remove a crashed or timed-out bot: its sites become neutral.
    pub fn eject(&mut self, player: usize) {
        for o in self.owner.iter_mut() {
            if *o as usize == player {
                *o = 0;
            }
        }
    }
}

/// Append the decimal form of `v` (much faster than `write!`).
pub fn push_int(out: &mut String, v: i64) {
    if v < 0 {
        out.push('-');
        push_int(out, -v);
        return;
    }
    let mut buf = [0u8; 20];
    let mut i = buf.len();
    let mut v = v as u64;
    loop {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    out.push_str(std::str::from_utf8(&buf[i..]).unwrap());
}
