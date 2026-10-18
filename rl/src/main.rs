//! Behaviour cloning trainer for the neural Halite bot (candle on Metal).
//!
//!     rl train  --data rl/data/train --val rl/data/val --run rl/runs/bc1 [--steps 20000] ...
//!     rl parity --run rl/runs/bc1 --val rl/data/val      candle forward vs the std-only nn.rs
//!
//! Every evaluation writes `model.safetensors` (to resume) and `model.hnn` (what the bot loads)
//! into the run directory. Features and the weight format live in starter-kit/rust/src/nn.rs,
//! shared with the bot.

mod metal_conv;
mod ppo;
mod sim;
#[allow(dead_code)]
#[path = "../../engine/src/game.rs"]
mod game;
#[allow(dead_code)]
#[path = "../../engine/src/pyrandom.rs"]
mod pyrandom;
#[path = "../../starter-kit/rust/src/nn.rs"]
mod nn;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{AdamW, Init, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Replays
// ---------------------------------------------------------------------------

struct Game {
    w: usize,
    h: usize,
    frames: usize,
    prod: Vec<u8>,
    owners: Vec<u8>,    // frames * w * h
    strengths: Vec<u8>, // frames * w * h
    moves: Vec<u8>,     // (frames - 1) * w * h, digits 0-4
    teachers: Vec<u8>,  // player tags whose moves we imitate
}

fn load_game(path: &Path, teacher: &str) -> Option<Game> {
    let bytes = std::fs::read(path).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let w = v["width"].as_u64()? as usize;
    let h = v["height"].as_u64()? as usize;
    let ints = |a: &serde_json::Value| -> Option<Vec<u8>> { a.as_array()?.iter().map(|x| x.as_u64().map(|x| x as u8)).collect() };
    let prod = ints(&v["productions"])?;
    let (mut owners, mut strengths) = (Vec::new(), Vec::new());
    let fr = v["frames"].as_array()?;
    for f in fr {
        owners.extend(ints(&f[0])?);
        strengths.extend(ints(&f[1])?);
    }
    let mut moves = Vec::new();
    for m in v["moves"].as_array()? {
        moves.extend(m.as_str()?.bytes().map(|b| b - b'0'));
    }
    let teachers = v["player_names"].as_array()?.iter().enumerate()
        .filter(|(_, n)| n.as_str() == Some(teacher)).map(|(i, _)| i as u8 + 1).collect();
    Some(Game { w, h, frames: fr.len(), prod, owners, strengths, moves, teachers })
}

fn load_dir(dir: &str, teacher: &str, max_games: usize) -> Vec<Game> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir).expect("data dir").filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |x| x == "hlt")).collect();
    paths.sort();
    paths.truncate(max_games);
    let chunks: Vec<Vec<PathBuf>> = paths.chunks(paths.len().div_ceil(8).max(1)).map(|c| c.to_vec()).collect();
    let mut games = Vec::new();
    std::thread::scope(|s| {
        let hs: Vec<_> = chunks.iter().map(|c| s.spawn(move || c.iter().filter_map(|p| load_game(p, teacher)).collect::<Vec<_>>())).collect();
        for hd in hs {
            games.extend(hd.join().unwrap());
        }
    });
    games
}

#[derive(Clone, Copy)]
struct Sample {
    game: u32,
    frame: u16,
    player: u8,
}

/// Every (frame, teacher seat) where the teacher owns at least one site, bucketed by map size.
fn index(games: &[Game]) -> HashMap<(usize, usize), Vec<Sample>> {
    let mut b: HashMap<(usize, usize), Vec<Sample>> = HashMap::new();
    for (gi, g) in games.iter().enumerate() {
        let n = g.w * g.h;
        for f in 0..g.frames - 1 {
            let own = &g.owners[f * n..(f + 1) * n];
            for &p in &g.teachers {
                if own.contains(&p) {
                    b.entry((g.w, g.h)).or_default().push(Sample { game: gi as u32, frame: f as u16, player: p });
                }
            }
        }
    }
    b
}

// ---------------------------------------------------------------------------
// Batches, with the 8 symmetries of the torus as augmentation
// ---------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// sym bits: 4 transpose, 1 flip x, 2 flip y (applied in that order).
fn sym_dims(w: usize, h: usize, sym: usize) -> (usize, usize) {
    if sym & 4 != 0 { (h, w) } else { (w, h) }
}
fn sym_site(w: usize, h: usize, sym: usize, x: usize, y: usize) -> usize {
    let (w2, h2) = sym_dims(w, h, sym);
    let (mut a, mut b) = if sym & 4 != 0 { (y, x) } else { (x, y) };
    if sym & 1 != 0 { a = w2 - 1 - a; }
    if sym & 2 != 0 { b = h2 - 1 - b; }
    b * w2 + a
}
fn sym_dir(d: u8, sym: usize) -> u8 {
    let (mut dx, mut dy): (i32, i32) = match d { 1 => (0, -1), 2 => (1, 0), 3 => (0, 1), 4 => (-1, 0), _ => return 0 };
    if sym & 4 != 0 { std::mem::swap(&mut dx, &mut dy); }
    if sym & 1 != 0 { dx = -dx; }
    if sym & 2 != 0 { dy = -dy; }
    match (dx, dy) { (0, -1) => 1, (1, 0) => 2, (0, 1) => 3, _ => 4 }
}

struct Batch {
    b: usize,
    w: usize,
    h: usize,
    x: Vec<f32>,   // b*h*w*F
    tgt: Vec<u32>, // b*h*w
    mask: Vec<f32>,
}

fn make_batch(games: &[Game], samples: &[Sample], sym: usize) -> Batch {
    let g0 = &games[samples[0].game as usize];
    let (w0, h0) = (g0.w, g0.h);
    let (w, h) = sym_dims(w0, h0, sym);
    let n = w * h;
    let nf = nn::NUM_FEATURES;
    let mut x = vec![0f32; samples.len() * n * nf];
    let mut tgt = vec![0u32; samples.len() * n];
    let mut mask = vec![0f32; samples.len() * n];
    for (bi, s) in samples.iter().enumerate() {
        let g = &games[s.game as usize];
        let f = s.frame as usize;
        let own = &g.owners[f * n..(f + 1) * n];
        let st = &g.strengths[f * n..(f + 1) * n];
        let mv = &g.moves[f * n..(f + 1) * n];
        let feat = nn::features(w0, h0, own, st, &g.prod, s.player, f);
        // every frame weighs the same in the loss, however many sites the player owns: otherwise
        // the opening (one site) is drowned by mid-game frames with hundreds of sites
        let weight = 1.0 / own.iter().filter(|&&o| o == s.player).count() as f32;
        for y in 0..h0 {
            for xx in 0..w0 {
                let i = y * w0 + xx;
                let j = bi * n + sym_site(w0, h0, sym, xx, y);
                x[j * nf..(j + 1) * nf].copy_from_slice(&feat[i * nf..(i + 1) * nf]);
                if own[i] == s.player {
                    mask[j] = weight;
                    tgt[j] = sym_dir(mv[i], sym) as u32;
                }
            }
        }
    }
    Batch { b: samples.len(), w, h, x, tgt, mask }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

struct Model {
    channels: usize,
    dil: Vec<usize>,
    stem_w: Tensor,
    stem_b: Tensor,
    blocks: Vec<(Tensor, Tensor, Tensor)>,
    head_w: Tensor,
    head_b: Tensor,
    /// value head (RL), a small MLP on the *detached* trunk (value learning cannot disturb the
    /// policy): [mean of h over the map, mean of h over our sites, map means of the 11 input
    /// planes, of mine x production and of enemy x production] -> 64 -> tanh
    value_w1: Tensor,
    value_b1: Tensor,
    value_w2: Tensor,
    value_b2: Tensor,
}

/// Loads the tensors of a safetensors file into the matching variables; returns the names of
/// variables the file does not have (e.g. the value head when starting RL from a cloned network).
fn load_partial(varmap: &VarMap, path: &Path, dev: &Device) -> Result<Vec<String>> {
    let file = candle_core::safetensors::load(path, dev)?;
    let mut missing = Vec::new();
    for (name, var) in varmap.data().lock().unwrap().iter() {
        match file.get(name) {
            Some(t) => var.set(t)?,
            None => missing.push(name.clone()),
        }
    }
    missing.sort();
    Ok(missing)
}

/// Use the hand-written Metal conv (metal_conv.rs) instead of gather + matmul (`--conv gather`).
static USE_KERNEL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static FAKE_CONV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Run each layer as one fused Metal op (metal_conv::FusedConv). Implies the Metal conv.
static FUSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

thread_local! {
    /// im2col gather indices per (h, w, dilation): for each site its 9 taps, row-major.
    static TAPS: std::cell::RefCell<HashMap<(usize, usize, usize), Tensor>> = std::cell::RefCell::new(HashMap::new());
}

fn taps(h: usize, w: usize, d: usize, dev: &Device) -> Result<Tensor> {
    if let Some(t) = TAPS.with(|m| m.borrow().get(&(h, w, d)).cloned()) {
        return Ok(t);
    }
    let mut idx = Vec::with_capacity(h * w * 9);
    for y in 0..h {
        for x in 0..w {
            for ky in 0..3 {
                for kx in 0..3 {
                    let sy = (y + h * d + ky * d - d) % h;
                    let sx = (x + w * d + kx * d - d) % w;
                    idx.push((sy * w + sx) as u32);
                }
            }
        }
    }
    let t = Tensor::from_vec(idx, h * w * 9, dev)?;
    TAPS.with(|m| m.borrow_mut().insert((h, w, d), t.clone()));
    Ok(t)
}

/// Wrap-around 3x3 convolution, channel-last: x (B,H,W,Cin), w (9*Cin, Cout). Tap order as in nn.rs.
/// One gather (im2col) and one matmul: candle's backward of a gather is a single scatter-add.
fn conv(x: &Tensor, w: &Tensor, b: &Tensor, d: usize) -> Result<Tensor> {
    if FAKE_CONV.load(std::sync::atomic::Ordering::Relaxed) {
        // profiling only: a 1x1 stand-in (centre tap) to measure everything but the convolution
        let (bs, h, wd, c) = x.dims4()?;
        return x.reshape((bs * h * wd, c))?.matmul(&w.narrow(0, 4 * c, c)?)?.broadcast_add(b)?.reshape((bs, h, wd, ()));
    }
    if USE_KERNEL.load(std::sync::atomic::Ordering::Relaxed) && x.device().is_metal() {
        return x.contiguous()?.apply_op2(w, metal_conv::WrapConv { d, residual: false })?.broadcast_add(b);
    }
    let (bs, h, wd, c) = x.dims4()?;
    let col = x.reshape((bs, h * wd, c))?.index_select(&taps(h, wd, d, x.device())?, 1)?.reshape((bs * h * wd, 9 * c))?;
    col.matmul(w)?.broadcast_add(b)?.reshape((bs, h, wd, ()))
}

impl Model {
    fn new(vb: &VarBuilder, channels: usize, dil: &[usize]) -> Result<Model> {
        let c = channels;
        let nf = nn::NUM_FEATURES;
        let he = |fan: usize, scale: f64| Init::Randn { mean: 0.0, stdev: scale * (2.0 / fan as f64).sqrt() };
        let zero = Init::Const(0.0);
        let mut blocks = Vec::new();
        for k in 0..dil.len() {
            blocks.push((
                vb.get_with_hints((9 * c, c), &format!("block{k}.w"), he(9 * c, 0.3))?,
                vb.get_with_hints(c, &format!("block{k}.b"), zero)?,
                vb.get_with_hints((c, c), &format!("block{k}.g"), he(c, 0.1))?,
            ));
        }
        Ok(Model {
            channels,
            dil: dil.to_vec(),
            stem_w: vb.get_with_hints((9 * nf, c), "stem.w", he(9 * nf, 1.0))?,
            stem_b: vb.get_with_hints(c, "stem.b", zero)?,
            blocks,
            head_w: vb.get_with_hints((c, nn::NUM_MOVES), "head.w", he(c, 0.1))?,
            head_b: vb.get_with_hints(nn::NUM_MOVES, "head.b", zero)?,
            value_w1: vb.get_with_hints((2 * c + nf + 2, 64), "value.w1", he(2 * c + nf + 2, 1.0))?,
            value_b1: vb.get_with_hints(64, "value.b1", zero)?,
            value_w2: vb.get_with_hints((64, 1), "value.w2", he(64, 0.1))?,
            value_b2: vb.get_with_hints(1, "value.b2", zero)?,
        })
    }

    /// x (B,H,W,F) -> logits (B*H*W, 5)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (bs, h, w, _) = x.dims4()?;
        self.trunk(x)?.reshape((bs * h * w, self.channels))?.matmul(&self.head_w)?.broadcast_add(&self.head_b)
    }

    /// x (B,H,W,F) -> (logits (B*H*W, 5), value (B)). Feature 0 marks our sites.
    fn forward_pv(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (bs, h, w, _) = x.dims4()?;
        let c = self.channels;
        let hid = self.trunk(x)?.reshape((bs, h * w, c))?;
        let logits = hid.reshape((bs * h * w, c))?.matmul(&self.head_w)?.broadcast_add(&self.head_b)?;
        let hd = hid.detach();
        let xf = x.reshape((bs, h * w, nn::NUM_FEATURES))?;
        let mine = xf.narrow(2, 0, 1)?;
        let count = (metal_conv::pixel_mean(&mine)? * (h * w) as f64)?.clamp(1.0, f64::MAX)?;
        let mean_mine = (metal_conv::pixel_mean(&hd.broadcast_mul(&mine)?)? * (h * w) as f64)?.broadcast_div(&count)?;
        let prod = xf.narrow(2, 4, 1)?;
        let my_prod = (&mine * &prod)?;
        let enemy_prod = (xf.narrow(2, 1, 1)? * &prod)?;
        // one pass over the pixels for all the map means the value head reads
        let means = metal_conv::pixel_mean(&Tensor::cat(&[&hd, &xf, &my_prod, &enemy_prod], 2)?)?;
        let feats = Tensor::cat(&[means.narrow(1, 0, c)?, mean_mine, means.narrow(1, c, nn::NUM_FEATURES + 2)?], 1)?;
        let hv = feats.matmul(&self.value_w1)?.broadcast_add(&self.value_b1)?.relu()?;
        let value = hv.matmul(&self.value_w2)?.broadcast_add(&self.value_b2)?.tanh()?.squeeze(1)?;
        Ok((logits, value))
    }

    /// The shared body: x (B,H,W,F) -> h (B,H,W,C)
    fn trunk(&self, x: &Tensor) -> Result<Tensor> {
        let (bs, h, w, _) = x.dims4()?;
        let c = self.channels;
        if FUSED.load(std::sync::atomic::Ordering::Relaxed) && x.device().is_metal() && !FAKE_CONV.load(std::sync::atomic::Ordering::Relaxed) {
            // same function as below, one fused op per layer
            let gb = self.stem_b.unsqueeze(0)?.broadcast_as((bs, c))?.contiguous()?;
            let mut hcur = x.contiguous()?.apply_op3(&self.stem_w, &gb, metal_conv::FusedConv { d: 1, residual: false, input_grad: false, global: None })?;
            for ((wt, b, g), &d) in self.blocks.iter().zip(&self.dil) {
                // the mean is detached: FusedConv's backward adds its gradient path itself
                let gb = metal_conv::pixel_mean(&hcur.reshape((bs, h * w, c))?)?.detach().matmul(g)?.broadcast_add(b)?.contiguous()?;
                let op = metal_conv::FusedConv { d, residual: true, input_grad: true, global: Some(g.detach()) };
                hcur = hcur.apply_op3(wt, &gb, op)?;
            }
            return Ok(hcur);
        }
        let mut hcur = conv(x, &self.stem_w, &self.stem_b, 1)?.relu()?;
        for ((wt, b, g), &d) in self.blocks.iter().zip(&self.dil) {
            let glob = hcur.mean_keepdim(1)?.mean_keepdim(2)?.reshape((bs, c))?.matmul(g)?.reshape((bs, 1, 1, c))?;
            let t = conv(&hcur, wt, b, d)?;
            hcur = (hcur + t)?.broadcast_add(&glob)?.relu()?;
        }
        Ok(hcur)
    }

    fn export(&self) -> Result<Vec<u8>> {
        let mut ts: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        let mut arch = vec![nn::NUM_FEATURES as f32, self.channels as f32];
        arch.extend(self.dil.iter().map(|&d| d as f32));
        ts.push(("arch".into(), vec![arch.len()], arch));
        let mut put = |name: String, t: &Tensor| -> Result<()> {
            ts.push((name, t.dims().to_vec(), t.flatten_all()?.to_vec1::<f32>()?));
            Ok(())
        };
        put("stem.w".into(), &self.stem_w)?;
        put("stem.b".into(), &self.stem_b)?;
        for (k, (w, b, g)) in self.blocks.iter().enumerate() {
            put(format!("block{k}.w"), w)?;
            put(format!("block{k}.b"), b)?;
            put(format!("block{k}.g"), g)?;
        }
        put("head.w".into(), &self.head_w)?;
        put("head.b".into(), &self.head_b)?;
        Ok(nn::write_tensors(&ts))
    }
}

/// (weighted NLL sum, [sum of weights (= frames), sites, correct, teacher moves that are not STILL, of those correct])
fn batch_stats(model: &Model, batch: &Batch, dev: &Device) -> Result<(Tensor, [f64; 5])> {
    let x = Tensor::from_slice(&batch.x, (batch.b, batch.h, batch.w, nn::NUM_FEATURES), dev)?;
    let tgt = Tensor::from_slice(&batch.tgt, batch.tgt.len(), dev)?;
    let mask = Tensor::from_slice(&batch.mask, batch.mask.len(), dev)?;
    let logits = model.forward(&x)?;
    let logp = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
    let nll = logp.gather(&tgt.unsqueeze(1)?, 1)?.squeeze(1)?.neg()?;
    let sum = (nll * &mask)?.sum_all()?;
    let wsum = batch.mask.iter().sum::<f32>() as f64;
    let mut cnt = 0.0;
    let pred: Vec<u32> = logits.argmax(D::Minus1)?.to_vec1()?;
    let (mut ok, mut mv, mut mv_ok) = (0.0, 0.0, 0.0);
    for i in 0..pred.len() {
        if batch.mask[i] > 0.0 {
            cnt += 1.0;
            ok += (pred[i] == batch.tgt[i]) as u8 as f64;
            if batch.tgt[i] != 0 {
                mv += 1.0;
                mv_ok += (pred[i] == batch.tgt[i]) as u8 as f64;
            }
        }
    }
    Ok((sum, [wsum, cnt, ok, mv, mv_ok]))
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

fn args() -> HashMap<String, String> {
    let a: Vec<String> = std::env::args().skip(2).collect();
    let mut m = HashMap::new();
    let mut i = 0;
    while i < a.len() {
        let k = a[i].trim_start_matches("--").to_string();
        let v = a.get(i + 1).cloned().unwrap_or_default();
        m.insert(k, v);
        i += 2;
    }
    m
}

fn val_batches(games: &[Game], pixels: usize, max_samples: usize) -> Vec<Batch> {
    let idx = index(games);
    let mut keys: Vec<_> = idx.keys().cloned().collect();
    keys.sort();
    let total: usize = idx.values().map(|v| v.len()).sum();
    let mut out = Vec::new();
    for k in keys {
        let v = &idx[&k];
        // an even stride over each bucket, so validation is fixed and covers whole games
        let take = (v.len() * max_samples).div_ceil(total).max(1);
        let picked: Vec<Sample> = (0..take).map(|i| v[i * v.len() / take]).collect();
        let per = (pixels / (k.0 * k.1)).max(1);
        for ch in picked.chunks(per) {
            out.push(make_batch(games, ch, 0));
        }
    }
    out
}

fn evaluate(model: &Model, batches: &[Batch], dev: &Device) -> Result<[f64; 4]> {
    let (mut nll, mut acc) = (0.0, [0.0; 5]);
    for b in batches {
        let (s, st) = batch_stats(model, b, dev)?;
        nll += s.to_scalar::<f32>()? as f64;
        for i in 0..5 {
            acc[i] += st[i];
        }
    }
    Ok([nll / acc[0], acc[2] / acc[1], acc[4] / acc[3].max(1.0), acc[1]])
}

fn train() -> Result<()> {
    let a = args();
    let get = |k: &str, d: &str| a.get(k).cloned().unwrap_or(d.to_string());
    let run = PathBuf::from(get("run", "rl/runs/bc"));
    std::fs::create_dir_all(&run).unwrap();
    let teacher = get("teacher", "DexBot");
    let steps: usize = get("steps", "20000").parse().unwrap();
    let pixels: usize = get("pixels", "65536").parse().unwrap();
    let lr: f64 = get("lr", "1e-3").parse().unwrap();
    let channels: usize = get("channels", "32").parse().unwrap();
    let dil: Vec<usize> = get("dil", "1,1,2,1,2,4,1,2,4,8").split(',').map(|s| s.parse().unwrap()).collect();
    let eval_every: usize = get("eval-every", "1000").parse().unwrap();
    let max_games: usize = get("max-games", "1000000").parse().unwrap();
    let seed: u64 = get("seed", "1").parse().unwrap();
    let conv_impl = get("conv", "fused");
    USE_KERNEL.store(conv_impl != "gather", std::sync::atomic::Ordering::Relaxed);
    FUSED.store(conv_impl == "fused", std::sync::atomic::Ordering::Relaxed);

    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(run.join("log.txt")).unwrap();
    let mut say = |s: String| {
        println!("{s}");
        writeln!(log, "{s}").unwrap();
    };
    say(format!("args: {:?}", std::env::args().collect::<Vec<_>>()));

    let t0 = Instant::now();
    let games = std::sync::Arc::new(load_dir(&get("data", "rl/data/train"), &teacher, max_games));
    let val_games = load_dir(&get("val", "rl/data/val"), &teacher, 1000000);
    let buckets = index(&games);
    let mut keys: Vec<(usize, usize)> = buckets.keys().cloned().collect();
    keys.sort();
    // pick a bucket in proportion to its pixel count, so every site is equally likely
    let weights: Vec<f64> = keys.iter().map(|k| (buckets[k].len() * k.0 * k.1) as f64).collect();
    let total_w: f64 = weights.iter().sum();
    let nsamples: usize = buckets.values().map(|v| v.len()).sum();
    let val = val_batches(&val_games, pixels, 3000);
    say(format!("{} games, {} samples, {} sizes, {} val batches, loaded in {:.1?}", games.len(), nsamples, keys.len(), val.len(), t0.elapsed()));

    // batches are built on two CPU threads while the GPU trains
    let (tx, rx) = mpsc::sync_channel::<Batch>(8);
    let buckets = std::sync::Arc::new(buckets);
    for t in 0..2u64 {
        let (tx, games, buckets, keys, weights) = (tx.clone(), games.clone(), buckets.clone(), keys.clone(), weights.clone());
        std::thread::spawn(move || {
            let mut rng = Rng(seed * 1000 + t);
            loop {
                let mut r = (rng.next() >> 11) as f64 / (1u64 << 53) as f64 * total_w;
                let mut ki = 0;
                while ki + 1 < keys.len() && r >= weights[ki] {
                    r -= weights[ki];
                    ki += 1;
                }
                let k = keys[ki];
                let v = &buckets[&k];
                let per = (pixels / (k.0 * k.1)).max(1);
                let ss: Vec<Sample> = (0..per).map(|_| v[rng.below(v.len())]).collect();
                if tx.send(make_batch(&games, &ss, rng.below(8))).is_err() {
                    return;
                }
            }
        });
    }

    let dev = Device::new_metal(0)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = Model::new(&vb, channels, &dil)?;
    let mut start = 0;
    if run.join("model.safetensors").exists() && a.contains_key("resume") {
        load_partial(&varmap, &run.join("model.safetensors"), &dev)?;
        start = get("resume", "0").parse().unwrap();
        say(format!("resumed at step {start}"));
    }
    let nparams: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    say(format!("model: {channels} channels, dilations {dil:?}, {nparams} parameters"));
    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr, weight_decay: 1e-4, ..Default::default() })?;

    let warmup = 300.min(steps / 10);
    let (mut l_sum, mut l_n, mut s_ok, mut s_n) = (0.0, 0, 0.0, 0.0);
    // on resume, best.hnn keeps the lowest validation loss logged so far
    let mut best = std::fs::read_to_string(run.join("log.txt")).unwrap_or_default().lines()
        .filter_map(|l| l.strip_prefix("eval step ")?.split("val loss ").nth(1)?.split_whitespace().next()?.parse::<f64>().ok())
        .filter(|_| start > 0)
        .fold(f64::INFINITY, f64::min);
    let t1 = Instant::now();
    let mut wait = 0.0;
    for step in start..steps {
        let cur = if step < warmup { lr * (step + 1) as f64 / warmup as f64 } else {
            let p = (step - warmup) as f64 / (steps - warmup).max(1) as f64;
            lr * (0.02 + 0.98 * 0.5 * (1.0 + (std::f64::consts::PI * p).cos()))
        };
        opt.set_learning_rate(cur);
        let tw = Instant::now();
        let batch = rx.recv().unwrap();
        wait += tw.elapsed().as_secs_f64();
        let (sum, st) = batch_stats(&model, &batch, &dev)?;
        let loss = (sum / st[0])?;
        opt.backward_step(&loss)?;
        l_sum += loss.to_scalar::<f32>()? as f64;
        l_n += 1;
        s_ok += st[2];
        s_n += st[1];
        if (step + 1) % 100 == 0 {
            say(format!("step {:6}  loss {:.4}  acc {:.3}  lr {:.2e}  {:.0} steps/min  data wait {:.0}%", step + 1, l_sum / l_n as f64, s_ok / s_n, cur,
                (step + 1 - start) as f64 / t1.elapsed().as_secs_f64() * 60.0, 100.0 * wait / t1.elapsed().as_secs_f64()));
            (l_sum, l_n, s_ok, s_n) = (0.0, 0, 0.0, 0.0);
        }
        if (step + 1) % eval_every == 0 || step + 1 == steps {
            let [vl, va, vm, n] = evaluate(&model, &val, &dev)?;
            say(format!("eval step {}  val loss {:.4}  acc {:.4}  move recall {:.4}  ({} sites)", step + 1, vl, va, vm, n));
            varmap.save(run.join("model.safetensors"))?;
            std::fs::write(run.join("model.hnn"), model.export()?).unwrap();
            if vl < best {
                best = vl;
                std::fs::copy(run.join("model.hnn"), run.join("best.hnn")).unwrap();
            }
        }
    }
    Ok(())
}

fn parity() -> Result<()> {
    let a = args();
    let run = PathBuf::from(a.get("run").expect("--run"));
    let net = nn::Net::load(&std::fs::read(run.join("model.hnn")).unwrap()).unwrap();
    let tensors = nn::read_tensors(&std::fs::read(run.join("model.hnn")).unwrap()).unwrap();
    let arch = &tensors.iter().find(|t| t.0 == "arch").unwrap().2;
    let dil: Vec<usize> = arch[2..].iter().map(|&d| d as usize).collect();
    let dev = Device::new_metal(0)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = Model::new(&vb, arch[1] as usize, &dil)?;
    load_partial(&varmap, &run.join("model.safetensors"), &dev)?;
    let games = load_dir(a.get("val").map(|s| s.as_str()).unwrap_or("rl/data/val"), "DexBot", 5);
    let mut worst = 0f32;
    let mut agree = (0, 0);
    for (gi, g) in games.iter().enumerate() {
        let f = g.frames / 2;
        let s = Sample { game: gi as u32, frame: f as u16, player: g.teachers[0] };
        let b = make_batch(&games, &[s], 0);
        let x = Tensor::from_slice(&b.x, (1, b.h, b.w, nn::NUM_FEATURES), &dev)?;
        let gpu: Vec<f32> = model.forward(&x)?.flatten_all()?.to_vec1()?;
        let t = Instant::now();
        let cpu = net.forward(b.w, b.h, &b.x);
        let el = t.elapsed();
        for i in 0..b.w * b.h {
            let am = |v: &[f32]| (0..5).max_by(|&p, &q| v[p].total_cmp(&v[q])).unwrap();
            agree.1 += 1;
            agree.0 += (am(&gpu[i * 5..i * 5 + 5]) == am(&cpu[i * 5..i * 5 + 5])) as usize;
        }
        let d = gpu.iter().zip(&cpu).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
        worst = worst.max(d);
        println!("game {gi} {}x{}: max |logit diff| {d:.2e}, std-only forward {el:.1?}", b.w, b.h);
    }
    println!("worst {worst:.2e}, argmax agreement {}/{}", agree.0, agree.1);
    Ok(())
}

/// Checks the Metal conv against gather + matmul (forward and both gradients), then times its
/// two kernels at 50x50. Training speed itself shows in `train`'s steps/min.
fn bench() -> Result<()> {
    use std::sync::atomic::Ordering::Relaxed;
    let dev = Device::new_metal(0)?;
    for &(b, h, w, cin, cout, d) in &[(2usize, 7usize, 5usize, 3usize, 4usize, 1usize), (3, 12, 9, 11, 32, 2), (2, 20, 20, 32, 32, 8)] {
        let x = candle_core::Var::randn(0f32, 1., (b, h, w, cin), &dev)?;
        let wt = candle_core::Var::randn(0f32, 0.2, (9 * cin, cout), &dev)?;
        let bias = Tensor::zeros(cout, DType::F32, &dev)?;
        let probe = Tensor::randn(0f32, 1., (b, h, w, cout), &dev)?;
        let mut outs = Vec::new();
        for k in [false, true] {
            USE_KERNEL.store(k, Relaxed);
            let y = conv(x.as_tensor(), wt.as_tensor(), &bias, d)?;
            let g = (&y * &probe)?.sum_all()?.backward()?;
            outs.push([y, g.get(&x).unwrap().clone(), g.get(&wt).unwrap().clone()]);
        }
        let diff = |a: &Tensor, c: &Tensor| -> Result<f32> { (a - c)?.abs()?.max_all()?.to_scalar::<f32>() };
        let mag = |a: &Tensor| -> Result<f32> { a.abs()?.max_all()?.to_scalar::<f32>() };
        println!("b{b} {h}x{w} {cin}->{cout} d{d}: max diff y {:.1e}  dx {:.1e}  dw {:.1e}  (max |y| {:.1}, |dx| {:.1}, |dw| {:.1})",
            diff(&outs[0][0], &outs[1][0])?, diff(&outs[0][1], &outs[1][1])?, diff(&outs[0][2], &outs[1][2])?,
            mag(&outs[1][0])?, mag(&outs[1][1])?, mag(&outs[1][2])?);
    }
    // pixel_mean against candle's mean, odd sizes included
    for &(b, n, c) in &[(3usize, 7usize, 11usize), (26, 2500, 32), (5, 1225, 77), (2, 600, 256)] {
        let t = Tensor::randn(0f32, 1., (b, n, c), &dev)?;
        let d = (metal_conv::pixel_mean(&t)? - t.mean(1)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("pixel_mean ({b}, {n}, {c}) vs candle: max diff {d:.1e}");
    }
    // matrix-unit kernels against the scalar ones, same inputs
    {
        let sync = |t: &Tensor| -> Result<()> { t.flatten_all()?.narrow(0, 0, 1)?.to_vec1::<f32>().map(|_| ()) };
        for &(b, cin, cout) in &[(3usize, 11usize, 32usize), (26, 32, 32)] {
            let x = Tensor::randn(0f32, 1., (b, 50, 50, cin), &dev)?;
            let w = Tensor::randn(0f32, 0.1, (9 * cin, cout), &dev)?;
            let gy = Tensor::randn(0f32, 1., (b, 50, 50, cout), &dev)?;
            let mut res = Vec::new();
            for sg in [false, true] {
                metal_conv::MATRIX_UNITS.store(sg, Relaxed);
                let (mut tf, mut tw) = (f64::INFINITY, f64::INFINITY);
                let (mut y, mut g) = (None, None);
                for _ in 0..4 {
                    let t = Instant::now();
                    let yy = x.apply_op2_no_bwd(&w, &metal_conv::WrapConv { d: 2, residual: false })?;
                    sync(&yy)?;
                    tf = tf.min(t.elapsed().as_secs_f64());
                    let t = Instant::now();
                    let gg = metal_conv::wgrad(&x, &gy, 2)?;
                    sync(&gg)?;
                    tw = tw.min(t.elapsed().as_secs_f64());
                    (y, g) = (Some(yy), Some(gg));
                }
                println!("  b{b} {cin}->{cout} {}: fwd {:.2} ms, wgrad {:.2} ms", if sg { "matrix" } else { "scalar" }, tf * 1e3, tw * 1e3);
                res.push((y.unwrap(), g.unwrap()));
            }
            let d = |a: &Tensor, c: &Tensor| -> Result<f32> { (a - c)?.abs()?.max_all()?.to_scalar::<f32>() };
            println!("  b{b} {cin}->{cout}: matrix vs scalar max diff y {:.1e}, dw {:.1e}", d(&res[0].0, &res[1].0)?, d(&res[0].1, &res[1].1)?);
        }
        metal_conv::MATRIX_UNITS.store(true, Relaxed);
    }
    // whole model: fused layers against the reference (gather + matmul, candle ops), all gradients
    {
        let dil = [1, 2, 4, 8];
        let varmap = VarMap::new();
        let model = Model::new(&VarBuilder::from_varmap(&varmap, DType::F32, &dev), 32, &dil)?;
        let x = Tensor::rand(0f32, 1., (3, 13, 11, nn::NUM_FEATURES), &dev)?;
        let probe = Tensor::randn(0f32, 1., (3 * 13 * 11, nn::NUM_MOVES), &dev)?;
        let mut runs = Vec::new();
        for (kernel, fused, sg) in [(false, false, false), (true, true, false), (true, true, true)] {
            USE_KERNEL.store(kernel, Relaxed);
            FUSED.store(fused, Relaxed);
            metal_conv::MATRIX_UNITS.store(sg, Relaxed);
            let out = model.forward(&x)?;
            let g = (&out * &probe)?.sum_all()?.backward()?;
            // the value head gets no gradient from this policy-only probe
            let grads: Vec<Tensor> = varmap.all_vars().iter().filter_map(|v| g.get(v).cloned()).collect();
            runs.push((out, grads));
        }
        let rel = |a: &Tensor, c: &Tensor| -> Result<f32> {
            Ok((a - c)?.abs()?.max_all()?.to_scalar::<f32>()? / c.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-12))
        };
        for (k, name) in [(1, "fused, scalar kernels"), (2, "fused, matrix units")] {
            let mut worst = 0f32;
            for (a, c) in runs[0].1.iter().zip(&runs[k].1) {
                worst = worst.max(rel(a, c)?);
            }
            println!("model, {name} vs reference: logits rel diff {:.1e}, worst parameter gradient rel diff {:.1e} ({} tensors)",
                rel(&runs[k].0, &runs[0].0)?, worst, runs[0].1.len());
        }
        USE_KERNEL.store(true, Relaxed);
        FUSED.store(true, Relaxed);
    }
    {
        let sync = |t: &Tensor| -> Result<()> { t.flatten_all()?.narrow(0, 0, 1)?.to_vec1::<f32>().map(|_| ()) };
        for &b in &[1usize, 4, 26] {
            let x = Tensor::randn(0f32, 1., (b, 50, 50, 32), &dev)?;
            let w = Tensor::randn(0f32, 0.1, (288, 32), &dev)?;
            for i in 0..3 {
                let t = Instant::now();
                let y = x.apply_op2_no_bwd(&w, &metal_conv::WrapConv { d: 1, residual: false })?;
                sync(&y)?;
                let tf = t.elapsed();
                let t = Instant::now();
                let g = metal_conv::wgrad(&x, &y, 1)?;
                sync(&g)?;
                if i == 2 {
                    println!("b{b} 50x50 32->32: fwd {tf:?} wgrad {:?}", t.elapsed());
                }
            }
        }
    }
    // whole training step (forward, loss, backward, AdamW) with the real conv and with a 1x1 stand-in
    let dil = [1, 1, 2, 1, 2, 4, 1, 2, 4, 8];
    let varmap = VarMap::new();
    let model = Model::new(&VarBuilder::from_varmap(&varmap, DType::F32, &dev), 32, &dil)?;
    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW::default())?;
    let x = Tensor::randn(0f32, 1., (26, 50, 50, nn::NUM_FEATURES), &dev)?;
    for (name, fused, sg) in [("Metal conv, unfused, scalar", false, false), ("fused, scalar kernels", true, false), ("fused, matrix units", true, true)] {
        FUSED.store(fused, Relaxed);
        metal_conv::MATRIX_UNITS.store(sg, Relaxed);
            let mut best = f64::INFINITY;
        for _ in 0..6 {
            let t = Instant::now();
            let loss = model.forward(&x)?.sqr()?.mean_all()?;
            opt.backward_step(&loss)?;
            loss.to_scalar::<f32>()?;
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("full step, {name}: {:.1} ms", best * 1000.0);
    }
    FAKE_CONV.store(false, Relaxed);
    FUSED.store(true, Relaxed);
    // fused step split into its phases (each synced, so the sum exceeds a pipelined step a little)
    let sync = |t: &Tensor| -> Result<()> { t.flatten_all()?.narrow(0, 0, 1)?.to_vec1::<f32>().map(|_| ()) };
    let mut tt = [f64::INFINITY; 4];
    for _ in 0..6 {
        let t = Instant::now();
        let out = model.forward(&x)?;
        sync(&out)?;
        let t1 = t.elapsed().as_secs_f64();
        let loss = candle_nn::ops::log_softmax(&out, D::Minus1)?.mean_all()?;
        loss.to_scalar::<f32>()?;
        let t2 = t.elapsed().as_secs_f64();
        let g = loss.backward()?;
        for v in varmap.all_vars() {
            if let Some(gv) = g.get(&v) {
                sync(gv)?;
            }
        }
        let t3 = t.elapsed().as_secs_f64();
        opt.step(&g)?;
        sync(&varmap.all_vars()[0])?;
        let t4 = t.elapsed().as_secs_f64();
        for (k, v) in [t1, t2 - t1, t3 - t2, t4 - t3].into_iter().enumerate() {
            tt[k] = tt[k].min(v);
        }
    }
    {
        let t4 = Tensor::randn(0f32, 1., (26, 50, 50, 32), &dev)?;
        let part = Tensor::randn(0f32, 1., (128, 9216), &dev)?;
        let wq = Tensor::randn(0f32, 0.1, (288, 32), &dev)?;
        let ops: Vec<(&str, Box<dyn Fn() -> Result<Tensor>>)> = vec![
            ("sum over (h, w) of (26,50,50,32)", Box::new(|| t4.sum((1, 2)))),
            ("mean over pixels of (26,2500,32)", Box::new(|| t4.reshape((26, 2500, 32))?.mean(1))),
            ("pixel_mean kernel, same", Box::new(|| metal_conv::pixel_mean(&t4.reshape((26, 2500, 32))?))),
            ("sum(0) of wgrad chunks (128,9216)", Box::new(|| part.sum(0))),
            ("relu mask kernel", Box::new(|| t4.apply_op2_no_bwd(&t4, &metal_conv::ReluMask))),
            ("add two (26,50,50,32)", Box::new(|| &t4 + &t4)),
            ("broadcast (26,1,32) to (26,2500,32)", Box::new(|| t4.reshape((26, 2500, 32))?.narrow(1, 0, 1)?.broadcast_as((26, 2500, 32))?.contiguous())),
            ("conv kernel, residual", Box::new(|| t4.apply_op2_no_bwd(&wq, &metal_conv::WrapConv { d: 1, residual: true }))),
            ("wgrad", Box::new(|| metal_conv::wgrad(&t4, &t4, 1))),
        ];
        for (name, f) in &ops {
                let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t = Instant::now();
                sync(&f()?)?;
                best = best.min(t.elapsed().as_secs_f64());
            }
            println!("  {name}: {:.2} ms", best * 1e3);
        }
    }
    println!("fused step phases: forward {:.1} ms, loss {:.1} ms, backward {:.1} ms, AdamW {:.1} ms",
        tt[0] * 1e3, tt[1] * 1e3, tt[2] * 1e3, tt[3] * 1e3);
    Ok(())
}

fn main() -> Result<()> {
    match std::env::args().nth(1).as_deref() {
        Some("train") => train(),
        Some("parity") => parity(),
        Some("bench") => bench(),
        Some("rl") => ppo::run(),
        Some("simtest") => sim::check_replays(),
        _ => {
            eprintln!("usage: halite-rl train|parity|bench|rl [--key value ...] (see the top of src/main.rs and src/ppo.rs)");
            std::process::exit(2);
        }
    }
}
