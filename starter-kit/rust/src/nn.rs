//! Neural policy for Halite I: input features and a std-only forward pass.
//!
//! The network is trained by the `rl/` crate (candle, Metal), which includes this very file so
//! that features and inference cannot drift apart. Weights come in a small binary format written
//! by `rl export`:
//!
//!     "HNN1", u32 tensor count, then per tensor: u32 name length, name (UTF-8),
//!     u32 ndim, u32 dims.., f32 data (all little endian, row-major)
//!
//! Architecture (all convolutions wrap around the map edges, like the game):
//!     h = relu(conv3x3(x) + b)                                   stem, F -> C
//!     h = relu(h + conv3x3_d(h) + b + mean_xy(h) @ G)            one per block, dilation d
//!     logits = h @ W + b                                         5 moves per site
//! Tensors are channel-last ([y][x][c]); a 3x3 kernel is (9 * Cin, Cout) with tap
//! k = 3 * (dy + 1) + (dx + 1) reading the input at (y + dy * d, x + dx * d).

/// Planes of feature set v1 (NeuroAlpha's) and v2. A weights file records its plane count
/// (`arch[0]`), which selects the feature set: old weights keep working unchanged.
pub const NUM_FEATURES: usize = 11;
pub const NUM_FEATURES_V2: usize = 24;
pub const NUM_MOVES: usize = 5;

/// Features for a network that takes `planes` input planes (11: v1, 24: v2).
#[allow(clippy::too_many_arguments)]
pub fn features_for(planes: usize, w: usize, h: usize, owners: &[u8], strengths: &[u8], productions: &[u8], me: u8, turn: usize) -> Vec<f32> {
    if planes == NUM_FEATURES_V2 {
        features_v2(w, h, owners, strengths, productions, me, turn)
    } else {
        features(w, h, owners, strengths, productions, me, turn)
    }
}

/// Multi-source breadth-first distance on the wrap-around grid (4 neighbours), capped at `cap`.
fn bfs(w: usize, h: usize, source: impl Fn(usize) -> bool, cap: u16) -> Vec<u16> {
    let n = w * h;
    let mut d = vec![cap; n];
    let mut q = std::collections::VecDeque::new();
    for i in 0..n {
        if source(i) {
            d[i] = 0;
            q.push_back(i);
        }
    }
    while let Some(i) = q.pop_front() {
        if d[i] + 1 >= cap {
            continue;
        }
        let (x, y) = (i % w, i / w);
        for j in [((y + h - 1) % h) * w + x, ((y + 1) % h) * w + x, y * w + (x + 1) % w, y * w + (x + w - 1) % w] {
            if d[j] > d[i] + 1 {
                d[j] = d[i] + 1;
                q.push_back(j);
            }
        }
    }
    d
}

/// A potential field in the spirit of ErdmanBot's (his v26): every site we do not own starts at
/// strength / production (the turns a capture takes to pay back); stepping onto another unowned
/// site gives 0.9 x the neighbour's potential + 0.1 x its own (tunnelling past an expensive site
/// to a cheap one behind it), stepping into our land adds 0.2 x depth^2. Dijkstra order (the
/// updates never decrease a popped value). Low = attractive.
fn potential(w: usize, h: usize, owners: &[u8], strengths: &[u8], productions: &[u8], me: u8) -> Vec<f32> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let n = w * h;
    let seed: Vec<f32> = (0..n).map(|i| strengths[i] as f32 / (productions[i] as f32).max(0.5)).collect();
    let mut pot = vec![f32::INFINITY; n];
    let mut depth = vec![0u16; n];
    let mut heap = BinaryHeap::new();
    for i in 0..n {
        if owners[i] != me {
            pot[i] = seed[i];
            heap.push(Reverse((seed[i].to_bits(), i))); // non-negative floats order like their bits
        }
    }
    while let Some(Reverse((bits, i))) = heap.pop() {
        if f32::from_bits(bits) > pot[i] {
            continue;
        }
        let (x, y) = (i % w, i / w);
        for j in [((y + h - 1) % h) * w + x, ((y + 1) % h) * w + x, y * w + (x + 1) % w, y * w + (x + w - 1) % w] {
            let (cand, dj) = if owners[j] == me {
                let d = depth[i] + 1;
                (pot[i] + 0.2 * (d as f32) * (d as f32), d)
            } else {
                (0.9 * pot[i] + 0.1 * seed[j], 0)
            };
            if cand < pot[j] {
                pot[j] = cand;
                depth[j] = dj;
                heap.push(Reverse((cand.to_bits(), j)));
            }
        }
    }
    pot
}

/// Feature set v2: the 11 planes of v1, then
///   11 (x + y + turn) mod 2 (DexBot's chessboard)
///   12 distance to the nearest enemy site / 30 (1 when none)
///   13 our sites: depth inside our territory (distance to a site we do not own) / 30
///   14 other sites: distance to our territory / 30
///   15 neutral sites: production / max(strength, 5), capped at 1 (capture value)
///   16 potential field (see `potential`) as 1 / (1 + p / 5): high = attractive
///   17-19 my share of player-owned production, of the map's sites, of player-owned strength
///   20 enemy sites: their owner's total strength / (mine + theirs)
///   21 walls: neutral sites touching both us and an enemy
///   22 strongest enemy's total strength / (mine + theirs)
///   23 our own neighbours / 4
/// 24 planes, a multiple of 8, so the stem keeps the fast kernels.
#[allow(clippy::too_many_arguments)]
pub fn features_v2(w: usize, h: usize, owners: &[u8], strengths: &[u8], productions: &[u8], me: u8, turn: usize) -> Vec<f32> {
    let n = w * h;
    let v1 = features(w, h, owners, strengths, productions, me, turn);
    let nf = NUM_FEATURES_V2;
    let mut out = vec![0f32; n * nf];
    let (mut terr, mut strn, mut prod) = ([0f64; 256], [0f64; 256], [0f64; 256]);
    for i in 0..n {
        let o = owners[i] as usize;
        terr[o] += 1.0;
        strn[o] += strengths[i] as f64;
        prod[o] += productions[i] as f64;
    }
    let me_u = me as usize;
    let players_prod: f64 = prod[1..].iter().sum();
    let players_strn: f64 = strn[1..].iter().sum();
    let my_prod_share = (prod[me_u] / players_prod.max(1.0)) as f32;
    let my_terr_share = (terr[me_u] / n as f64) as f32;
    let my_strn_share = (strn[me_u] / players_strn.max(1.0)) as f32;
    let rel = |p: usize| (strn[p] / (strn[me_u] + strn[p]).max(1.0)) as f32;
    let strongest = (1..256).filter(|&p| p != me_u && terr[p] > 0.0).map(rel).fold(0f32, f32::max);
    let d_enemy = bfs(w, h, |i| owners[i] != 0 && owners[i] != me, 30);
    let d_out = bfs(w, h, |i| owners[i] != me, 30);
    let d_in = bfs(w, h, |i| owners[i] == me, 30);
    let pot = potential(w, h, owners, strengths, productions, me);
    for i in 0..n {
        let (x, y) = (i % w, i / w);
        let nb = [((y + h - 1) % h) * w + x, ((y + 1) % h) * w + x, y * w + (x + 1) % w, y * w + (x + w - 1) % w];
        let o = owners[i];
        let mine = o == me;
        let f = &mut out[i * nf..(i + 1) * nf];
        f[..NUM_FEATURES].copy_from_slice(&v1[i * NUM_FEATURES..(i + 1) * NUM_FEATURES]);
        f[11] = ((x + y + turn) % 2) as f32;
        f[12] = d_enemy[i] as f32 / 30.0;
        f[13] = if mine { d_out[i] as f32 / 30.0 } else { 0.0 };
        f[14] = if mine { 0.0 } else { d_in[i] as f32 / 30.0 };
        f[15] = if o == 0 { (productions[i] as f32 / (strengths[i] as f32).max(5.0)).min(1.0) } else { 0.0 };
        f[16] = 1.0 / (1.0 + pot[i] / 5.0);
        f[17] = my_prod_share;
        f[18] = my_terr_share;
        f[19] = my_strn_share;
        f[20] = if o != 0 && !mine { rel(o as usize) } else { 0.0 };
        f[21] = (o == 0 && nb.iter().any(|&j| owners[j] == me) && nb.iter().any(|&j| owners[j] != 0 && owners[j] != me)) as u8 as f32;
        f[22] = strongest;
        f[23] = nb.iter().filter(|&&j| owners[j] == me).count() as f32 / 4.0;
    }
    out
}

/// Turn limit of a W x H game (engine rule).
pub fn turn_limit(w: usize, h: usize) -> usize {
    (10.0 * ((w * h) as f64).sqrt()) as usize
}

/// Per-site input features seen by player `me`, channel-last [y][x][f]. `owners`, `strengths`
/// and `productions` are row-major W*H. `turn` counts from 0 (the first map the bot sees).
pub fn features(w: usize, h: usize, owners: &[u8], strengths: &[u8], productions: &[u8], me: u8, turn: usize) -> Vec<f32> {
    let n = w * h;
    let mut seen = [false; 256];
    for &o in owners {
        seen[o as usize] = true;
    }
    let enemies = (1..256).filter(|&p| seen[p] && p != me as usize).count();
    let t = turn as f32 / turn_limit(w, h).max(1) as f32;
    let e = enemies as f32 / 5.0;
    let mut out = vec![0f32; n * NUM_FEATURES];
    for i in 0..n {
        let o = owners[i];
        let s = strengths[i] as f32 / 255.0;
        let mine = (o == me) as u8 as f32;
        let neutral = (o == 0) as u8 as f32;
        let enemy = 1.0 - mine - neutral;
        let f = &mut out[i * NUM_FEATURES..(i + 1) * NUM_FEATURES];
        f[0] = mine;
        f[1] = enemy;
        f[2] = neutral;
        f[3] = s;
        f[4] = productions[i] as f32 / 16.0;
        f[5] = mine * s;
        f[6] = enemy * s;
        f[7] = neutral * s;
        f[8] = neutral * (strengths[i] == 0) as u8 as f32;
        f[9] = t;
        f[10] = e;
    }
    out
}

struct Block {
    dilation: usize,
    w: Vec<f32>,
    b: Vec<f32>,
    g: Vec<f32>,
}

pub struct Net {
    pub channels: usize,
    /// input planes: NUM_FEATURES (v1) or NUM_FEATURES_V2
    pub num_features: usize,
    stem_w: Vec<f32>,
    stem_b: Vec<f32>,
    blocks: Vec<Block>,
    head_w: Vec<f32>,
    head_b: Vec<f32>,
}

fn read_u32(b: &[u8], pos: &mut usize) -> Result<u32, String> {
    let s = b.get(*pos..*pos + 4).ok_or("truncated weights")?;
    *pos += 4;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

/// Parses the "HNN1" format into (name, dims, data) triples.
pub fn read_tensors(b: &[u8]) -> Result<Vec<(String, Vec<usize>, Vec<f32>)>, String> {
    if b.get(..4) != Some(b"HNN1") {
        return Err("not an HNN1 weights file".into());
    }
    let mut pos = 4;
    let count = read_u32(b, &mut pos)?;
    let mut out = Vec::new();
    for _ in 0..count {
        let len = read_u32(b, &mut pos)? as usize;
        let name = String::from_utf8(b.get(pos..pos + len).ok_or("truncated weights")?.to_vec()).map_err(|e| e.to_string())?;
        pos += len;
        let ndim = read_u32(b, &mut pos)?;
        let mut dims = Vec::new();
        for _ in 0..ndim {
            dims.push(read_u32(b, &mut pos)? as usize);
        }
        let numel: usize = dims.iter().product();
        let raw = b.get(pos..pos + 4 * numel).ok_or("truncated weights")?;
        pos += 4 * numel;
        let data = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        out.push((name, dims, data));
    }
    Ok(out)
}

/// Writes tensors in the "HNN1" format.
pub fn write_tensors(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    let mut out = b"HNN1".to_vec();
    out.extend((tensors.len() as u32).to_le_bytes());
    for (name, dims, data) in tensors {
        out.extend((name.len() as u32).to_le_bytes());
        out.extend(name.as_bytes());
        out.extend((dims.len() as u32).to_le_bytes());
        for &d in dims {
            out.extend((d as u32).to_le_bytes());
        }
        for &v in data {
            out.extend(v.to_le_bytes());
        }
    }
    out
}

impl Net {
    pub fn load(bytes: &[u8]) -> Result<Net, String> {
        let tensors = read_tensors(bytes)?;
        let get = |name: &str| -> Result<Vec<f32>, String> {
            tensors.iter().find(|t| t.0 == name).map(|t| t.2.clone()).ok_or(format!("missing tensor {}", name))
        };
        // arch = [features, channels, dilation of each block..]
        let arch = get("arch")?;
        if arch.len() < 2 || ![NUM_FEATURES, NUM_FEATURES_V2].contains(&(arch[0] as usize)) {
            return Err("weights were trained with an unknown feature set".into());
        }
        let num_features = arch[0] as usize;
        let channels = arch[1] as usize;
        let mut blocks = Vec::new();
        for (k, &d) in arch[2..].iter().enumerate() {
            blocks.push(Block {
                dilation: d as usize,
                w: get(&format!("block{}.w", k))?,
                b: get(&format!("block{}.b", k))?,
                g: get(&format!("block{}.g", k))?,
            });
        }
        Ok(Net {
            channels,
            num_features,
            stem_w: get("stem.w")?,
            stem_b: get("stem.b")?,
            blocks,
            head_w: get("head.w")?,
            head_b: get("head.b")?,
        })
    }

    /// The input planes this network was trained with, for player `me` at `turn`.
    #[allow(clippy::too_many_arguments)]
    pub fn features(&self, w: usize, h: usize, owners: &[u8], strengths: &[u8], productions: &[u8], me: u8, turn: usize) -> Vec<f32> {
        features_for(self.num_features, w, h, owners, strengths, productions, me, turn)
    }

    /// Move logits for every site, [y][x][5].
    pub fn forward(&self, w: usize, h: usize, x: &[f32]) -> Vec<f32> {
        let c = self.channels;
        let n = w * h;
        let mut cur = vec![0f32; n * c];
        conv3x3(x, self.num_features, w, h, 1, &self.stem_w, &self.stem_b, c, &mut cur);
        relu(&mut cur);
        let mut next = vec![0f32; n * c];
        for blk in &self.blocks {
            conv3x3(&cur, c, w, h, blk.dilation, &blk.w, &blk.b, c, &mut next);
            let mut mean = vec![0f32; c];
            for i in 0..n {
                for (m, v) in mean.iter_mut().zip(&cur[i * c..(i + 1) * c]) {
                    *m += v;
                }
            }
            let mut glob = vec![0f32; c];
            for (k, m) in mean.iter().enumerate() {
                let m = m / n as f32;
                for (gv, wv) in glob.iter_mut().zip(&blk.g[k * c..(k + 1) * c]) {
                    *gv += m * wv;
                }
            }
            for i in 0..n {
                let o = &mut next[i * c..(i + 1) * c];
                for k in 0..c {
                    o[k] = (cur[i * c + k] + o[k] + glob[k]).max(0.0);
                }
            }
            std::mem::swap(&mut cur, &mut next);
        }
        let mut out = vec![0f32; n * NUM_MOVES];
        for i in 0..n {
            let o = &mut out[i * NUM_MOVES..(i + 1) * NUM_MOVES];
            o.copy_from_slice(&self.head_b);
            for (k, &v) in cur[i * c..(i + 1) * c].iter().enumerate() {
                if v != 0.0 {
                    for (ov, wv) in o.iter_mut().zip(&self.head_w[k * NUM_MOVES..(k + 1) * NUM_MOVES]) {
                        *ov += v * wv;
                    }
                }
            }
        }
        out
    }
}

fn relu(v: &mut [f32]) {
    for x in v {
        *x = x.max(0.0);
    }
}

/// Threads for the convolutions: NEURO_THREADS, else 4 for big networks (more than 48 channels),
/// else 1. Rows are split between threads, so the result is identical whatever the count.
fn threads(cout: usize) -> usize {
    std::env::var("NEURO_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(if cout > 48 { 4 } else { 1 }).max(1)
}

#[allow(clippy::too_many_arguments)]
fn conv3x3(inp: &[f32], cin: usize, w: usize, h: usize, d: usize, wt: &[f32], b: &[f32], cout: usize, out: &mut [f32]) {
    let t = threads(cout).min(h);
    if t <= 1 {
        return conv_rows(inp, cin, w, h, d, wt, b, cout, out, 0);
    }
    let rows = h.div_ceil(t);
    std::thread::scope(|s| {
        for (k, chunk) in out.chunks_mut(rows * w * cout).enumerate() {
            s.spawn(move || conv_rows(inp, cin, w, h, d, wt, b, cout, chunk, k * rows));
        }
    });
}

/// The rows of `out` starting at map row `y0` (out holds whole rows, channel-last).
#[allow(clippy::too_many_arguments)]
fn conv_rows(inp: &[f32], cin: usize, w: usize, h: usize, d: usize, wt: &[f32], b: &[f32], cout: usize, out: &mut [f32], y0: usize) {
    for y in y0..y0 + out.len() / (w * cout) {
        for x in 0..w {
            let o = &mut out[((y - y0) * w + x) * cout..((y - y0) * w + x + 1) * cout];
            o.copy_from_slice(b);
            for ky in 0..3 {
                let sy = (y + h * d + ky * d - d) % h;
                for kx in 0..3 {
                    let sx = (x + w * d + kx * d - d) % w;
                    let src = &inp[(sy * w + sx) * cin..(sy * w + sx + 1) * cin];
                    let tap = (ky * 3 + kx) * cin;
                    for (ci, &v) in src.iter().enumerate() {
                        if v != 0.0 {
                            let row = &wt[(tap + ci) * cout..(tap + ci + 1) * cout];
                            for (ov, wv) in o.iter_mut().zip(row) {
                                *ov += v * wv;
                            }
                        }
                    }
                }
            }
        }
    }
}
