//! Reinforcement learning: PPO self-play on in-process games (sim.rs), starting from a cloned net.
//!
//!     halite-rl rl --init rl/runs/bc3/model.safetensors --run rl/runs/ppo3 [--iters 300] ...
//!
//! Each iteration plays `--games` games at once. The games come in `--types` (map side, players)
//! groups so that every turn needs only a few batched forward passes on the GPU. A type is solo
//! (one player, cut at `--solo-turns`) with probability `--solo`. One seat per game is the learner;
//! every other seat is the learner too with probability `--self-share`, otherwise a frozen
//! opponent (the starting network, plus snapshots of the learner added every `--snapshot-every`
//! iterations) playing its most likely move, like the released bot.
//! The learner samples its moves from softmax(logits / --temp) (0.5: close to the bot's argmax;
//! the same temperature is used in the update, so the ratio is exact).
//!
//! Credit (`--credit local`, the default; `frame` is the old scheme):
//! - global, per frame: final rank (+1 first .. -1 last) at the seat's last turn plus `--shaping` x
//!   the change of its production share each turn (of player-owned production; of the whole map
//!   in solo games), GAE with the scalar value head. One number for every site of the frame.
//! - local, per site: u(j) = [site j is mine] x (production(j) + `--local-strength` x strength(j));
//!   the return of site i at turn t is the discounted (`--local-gamma`) future change of u summed
//!   over the (2R+1)^2 window around i (R = `--local-radius`), baseline = the value-map head.
//!   A site is judged mostly by what happens around it, not by the other few hundred moves.
//!   advantage(site) = local (normalized per batch) + `--global-weight` x global (normalized).
//!   Measured with --snr: the frame-only gradient had no detectable signal at 96 games per half.
//! PPO with a per-site clipped ratio, each frame weighing the same as in cloning, value and
//! value-map losses, an entropy bonus and `--teacher-kl` x KL(policy || starting network) per
//! site (without it the policy drifted and got worse, run ppo1). The first `--value-warmup`
//! iterations train only the value heads (detached trunk: they never move the policy).
//! `--snr 1` measures the policy-gradient signal once the warmup is done, then exits.
//! Writes log.txt ("iter ..." lines), model.safetensors and model.hnn (the bot's format) to --run.

use crate::sim::Env;
use crate::{load_partial, nn, Model, Rng};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

struct Policy {
    model: Model,
    _varmap: VarMap,
    name: String,
}

/// A learner frame kept for the update.
struct Kept {
    env: usize,
    turn: usize,
    player: u8,
    /// index of this step in its trajectory, to read the advantage after GAE
    traj: usize,
    step: usize,
    actions: Vec<u8>,
    old_logp: Vec<f32>,
}

/// One learner seat's trajectory: rewards and values per turn it played.
#[derive(Default)]
struct Traj {
    rewards: Vec<f32>,
    values: Vec<f32>,
    adv: Vec<f32>,
    ret: Vec<f32>,
}

struct Game {
    env: Env,
    prod: Vec<u8>,
    /// seat p (1..=n): None = learner, Some(k) = frozen opponent k
    seats: Vec<Option<usize>>,
    /// learner seats: trajectory index
    traj: Vec<Option<usize>>,
    share: Vec<f64>,
    /// every frame of the game (owners, strengths), for the local returns
    hist: Vec<(Vec<u8>, Vec<u8>)>,
}

/// Each player's share of production: of all player-owned production, or of the whole map in a
/// solo game (where the share of player-owned production is always 1).
fn prod_share(env: &Env) -> Vec<f64> {
    let p = env.production_owned();
    let total: i64 = if env.n == 1 { p.iter().sum() } else { p[1..].iter().sum() };
    p.iter().map(|&v| if total > 0 { v as f64 / total as f64 } else { 0.0 }).collect()
}

/// Samples from softmax(l / temp); returns the move and its log-probability.
fn softmax_sample(l: &[f32], temp: f32, rng: &mut Rng) -> (usize, f32) {
    let m = l.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = l.iter().map(|v| ((v - m) / temp).exp()).collect();
    let z: f32 = e.iter().sum();
    let mut r = (rng.next() >> 11) as f32 / (1u64 << 53) as f32 * z;
    let mut k = 0;
    while k + 1 < e.len() && r >= e[k] {
        r -= e[k];
        k += 1;
    }
    (k, (e[k] / z).max(1e-12).ln())
}

fn argmax(l: &[f32]) -> usize {
    (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b]).then(b.cmp(&a))).unwrap()
}

fn load_policy(path: &PathBuf, channels: usize, dil: &[usize], dev: &Device, name: String) -> Result<Policy> {
    let varmap = VarMap::new();
    let model = Model::new(&VarBuilder::from_varmap(&varmap, DType::F32, dev), channels, dil)?;
    load_partial(&varmap, path, dev)?;
    Ok(Policy { model, _varmap: varmap, name })
}

/// Sum over the (2r+1)^2 window around every site, wrapping around the map edges.
fn box_sum(v: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for dx in 0..2 * r + 1 {
                s += v[y * w + (x + w * r + dx - r) % w];
            }
            tmp[y * w + x] = s;
        }
    }
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for dy in 0..2 * r + 1 {
                s += tmp[((y + h * r + dy - r) % h) * w + x];
            }
            out[y * w + x] = s;
        }
    }
    out
}

struct LocalCfg {
    gamma: f32,
    radius: usize,
    strength: f32,
    scale: f32,
}

/// Local returns of player p for the turns in `need`: G_t = box_sum(D_t), D_t = du_t + gamma D_{t+1}
/// with u as in the module doc. Divided by `scale` to keep the value-map targets near 1.
fn local_returns(g: &Game, p: u8, need: &[usize], c: &LocalCfg) -> HashMap<usize, Vec<f32>> {
    let (w, h) = (g.env.width(), g.env.height());
    let n = w * h;
    let u = |t: usize| -> Vec<f32> {
        let (own, st) = &g.hist[t];
        (0..n).map(|j| if own[j] == p { g.prod[j] as f32 + c.strength * st[j] as f32 } else { 0.0 }).collect()
    };
    let mut out = HashMap::new();
    let last = g.hist.len() - 1;
    let first = *need.iter().min().unwrap();
    let mut d = vec![0f32; n];
    let mut next = u(last);
    for t in (first..last).rev() {
        let cur = u(t);
        for j in 0..n {
            d[j] = (next[j] - cur[j]) + c.gamma * d[j];
        }
        if need.contains(&t) {
            out.insert(t, box_sum(&d, w, h, c.radius).into_iter().map(|v| v / c.scale).collect());
        }
        next = cur;
    }
    out
}

/// Tensors of one minibatch of kept frames (all the same map size).
struct Batch {
    bs: usize,
    xt: Tensor,
    act: Tensor,
    old: Tensor,
    /// site weight: 1 / sites owned on our sites, 0 elsewhere (each frame weighs the same)
    sw: Tensor,
    /// normalized global advantage of the frame, on our sites
    gadv: Tensor,
    /// local return of the site, on our sites
    lret: Tensor,
    ret: Tensor,
}

struct Ctx<'a> {
    games: &'a [Game],
    kept: &'a [Kept],
    trajs: &'a [Traj],
    frames: &'a HashMap<(usize, usize), (Vec<u8>, Vec<u8>)>,
    local: &'a HashMap<(usize, usize, u8), Vec<f32>>,
    mean: f32,
    std: f32,
}

fn build_batch(c: &Ctx, b: &[usize], dev: &Device) -> Result<Batch> {
    let e0 = &c.games[c.kept[b[0]].env].env;
    let (w, h) = (e0.width(), e0.height());
    let n = w * h;
    let bs = b.len();
    let mut x = Vec::with_capacity(bs * n * nn::NUM_FEATURES);
    let (mut act, mut old, mut sw, mut gadv, mut lret) =
        (Vec::with_capacity(bs * n), Vec::with_capacity(bs * n), Vec::with_capacity(bs * n), Vec::with_capacity(bs * n), Vec::with_capacity(bs * n));
    let mut ret = Vec::with_capacity(bs);
    for &ki in b {
        let k = &c.kept[ki];
        let (own, st) = &c.frames[&(k.env, k.turn)];
        x.extend(nn::features(w, h, own, st, &c.games[k.env].prod, k.player, k.turn));
        let owned = own.iter().filter(|&&o| o == k.player).count().max(1) as f32;
        let adv = (c.trajs[k.traj].adv[k.step] - c.mean) / c.std;
        let lr = c.local.get(&(k.env, k.turn, k.player));
        for i in 0..n {
            let mine = own[i] == k.player;
            act.push(k.actions[i] as u32);
            old.push(k.old_logp[i]);
            sw.push(if mine { 1.0 / owned } else { 0.0 });
            gadv.push(if mine { adv } else { 0.0 });
            lret.push(match (mine, lr) {
                (true, Some(v)) => v[i],
                _ => 0.0,
            });
        }
        ret.push(c.trajs[k.traj].ret[k.step]);
    }
    Ok(Batch {
        bs,
        xt: Tensor::from_vec(x, (bs, h, w, nn::NUM_FEATURES), dev)?,
        act: Tensor::from_vec(act, bs * n, dev)?,
        old: Tensor::from_vec(old, bs * n, dev)?,
        sw: Tensor::from_vec(sw, bs * n, dev)?,
        gadv: Tensor::from_vec(gadv, bs * n, dev)?,
        lret: Tensor::from_vec(lret, bs * n, dev)?,
        ret: Tensor::from_vec(ret, bs, dev)?,
    })
}

/// Per-site advantage x site weight: local residual (return - value map, normalized over the
/// batch's own sites) + global_weight x the frame's global advantage. `vmap` is detached.
fn site_advantage(b: &Batch, vmap: &Tensor, local_w: f64, global_w: f64) -> Result<Tensor> {
    let mask = b.sw.gt(0.0)?.to_dtype(DType::F32)?;
    let mut a = (&b.gadv * global_w)?;
    if local_w > 0.0 {
        let cnt = mask.sum_all()?.to_scalar::<f32>()?.max(1.0) as f64;
        let resid = ((&b.lret - vmap)? * &mask)?;
        let mu = resid.sum_all()?.to_scalar::<f32>()? as f64 / cnt;
        let centered = ((&resid - mu)? * &mask)?;
        let sd = (centered.sqr()?.sum_all()?.to_scalar::<f32>()? as f64 / cnt).sqrt().max(1e-6);
        a = (a + (centered * (local_w / sd))?)?;
    }
    a * &b.sw
}

/// Splits the kept frames `idx` into minibatches of one map size and about `pixels` sites.
fn minibatches(c: &Ctx, idx: &[usize], pixels: usize, rng: Option<&mut Rng>) -> Vec<Vec<usize>> {
    let mut by: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for &ki in idx {
        let e = &c.games[c.kept[ki].env].env;
        by.entry((e.width(), e.height())).or_default().push(ki);
    }
    let mut keys: Vec<_> = by.keys().cloned().collect();
    keys.sort();
    let mut out = Vec::new();
    let mut rng = rng;
    for k in keys {
        let mut v = by.remove(&k).unwrap();
        if let Some(r) = rng.as_deref_mut() {
            for i in (1..v.len()).rev() {
                v.swap(i, r.below(i + 1));
            }
        }
        out.extend(v.chunks((pixels / (k.0 * k.1)).max(1)).map(|c| c.to_vec()));
    }
    if let Some(r) = rng {
        for i in (1..out.len()).rev() {
            out.swap(i, r.below(i + 1));
        }
    }
    out
}

pub fn run() -> Result<()> {
    let a = crate::args();
    let get = |k: &str, d: &str| a.get(k).cloned().unwrap_or(d.to_string());
    let num = |k: &str, d: &str| get(k, d).parse::<f64>().unwrap();
    let run = PathBuf::from(get("run", "rl/runs/ppo"));
    std::fs::create_dir_all(&run).unwrap();
    let init = PathBuf::from(get("init", "rl/runs/bc3/model.safetensors"));
    let iters = num("iters", "300") as usize;
    let n_games = num("games", "96") as usize;
    let n_types = num("types", "8") as usize;
    let keep = num("keep", "0.25");
    let epochs = num("epochs", "2") as usize;
    let lr = num("lr", "5e-5");
    let clip = num("clip", "0.2") as f32;
    let ent_coef = num("ent", "0.003");
    let vcoef = num("vcoef", "0.5");
    let gamma = num("gamma", "0.999") as f32;
    let lam = num("lam", "0.95") as f32;
    let pixels = num("pixels", "65536") as usize;
    let shaping = num("shaping", "1.0") as f32;
    let value_warmup = num("value-warmup", "3") as usize;
    let self_share = num("self-share", "0.5");
    let snapshot_every = num("snapshot-every", "25") as usize;
    let max_snapshots = num("max-snapshots", "3") as usize;
    let save_every = num("save-every", "5") as usize;
    let seed = num("seed", "1") as u64;
    let temp = num("temp", "0.5");
    let teacher_kl = num("teacher-kl", "0.1");
    let solo = num("solo", "0.0");
    let solo_turns = num("solo-turns", "150") as usize;
    let credit = get("credit", "local");
    let (local_w, global_w): (f64, f64) = if credit == "local" { (1.0, num("global-weight", "0.5")) } else { (0.0, 1.0) };
    let lc = LocalCfg {
        gamma: num("local-gamma", "0.9") as f32,
        radius: num("local-radius", "4") as usize,
        strength: num("local-strength", "0.05") as f32,
        scale: num("local-scale", "20") as f32,
    };
    let channels = 32;
    let dil = vec![1, 1, 2, 1, 2, 4, 1, 2, 4, 8];

    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(run.join("log.txt")).unwrap();
    let mut say = |s: String| {
        println!("{s}");
        writeln!(log, "{s}").unwrap();
    };
    say(format!("args: {:?}", std::env::args().collect::<Vec<_>>()));

    let dev = Device::new_metal(0)?;
    let varmap = VarMap::new();
    let model = Model::new(&VarBuilder::from_varmap(&varmap, DType::F32, &dev), channels, &dil)?;
    let resume = run.join("model.safetensors");
    let mut start = 0;
    if a.contains_key("resume") && resume.exists() {
        load_partial(&varmap, &resume, &dev)?;
        start = get("resume", "0").parse().unwrap();
        say(format!("resumed from {} at iteration {start}", resume.display()));
    } else {
        let missing = load_partial(&varmap, &init, &dev)?;
        say(format!("initialised from {} (fresh: {:?})", init.display(), missing));
    }
    let mut opponents = vec![load_policy(&init, channels, &dil, &dev, "start".into())?];
    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr, weight_decay: 0.0, ..Default::default() })?;
    let mut rng = Rng(seed * 7919 + start as u64);

    for it in start..iters {
        let t_iter = Instant::now();
        // ---------------- games ----------------
        let mut games: Vec<Game> = Vec::with_capacity(n_games);
        let types: Vec<(usize, usize)> = (0..n_types)
            .map(|_| {
                let side = [20usize, 25, 30, 35, 40, 45, 50][rng.below(7)];
                let solo_type = (rng.next() >> 11) as f64 / ((1u64 << 53) as f64) < solo;
                (side, if solo_type { 1 } else { [2usize, 2, 3, 4, 4, 5, 6][rng.below(7)] })
            })
            .collect();
        for g in 0..n_games {
            let (side, n) = types[g % n_types];
            let mut env = Env::new(rng.next() as u32, side, side, n);
            if n == 1 {
                env.limit = env.limit.min(solo_turns);
            }
            let prod = env.productions();
            let learner_seat = 1 + rng.below(n);
            let mut seats = vec![None; n + 1];
            for (p, seat) in seats.iter_mut().enumerate().skip(1) {
                if p != learner_seat && (rng.next() >> 11) as f64 / (1u64 << 53) as f64 >= self_share {
                    *seat = Some(rng.below(opponents.len()));
                }
            }
            let share = prod_share(&env);
            let hist = vec![env.frame()];
            games.push(Game { env, prod, seats, traj: vec![None; n + 1], share, hist });
        }
        let mut trajs: Vec<Traj> = Vec::new();
        for g in games.iter_mut() {
            for p in 1..=g.env.n {
                if g.seats[p].is_none() {
                    g.traj[p] = Some(trajs.len());
                    trajs.push(Traj::default());
                }
            }
        }
        let mut kept: Vec<Kept> = Vec::new();
        let mut frames: HashMap<(usize, usize), (Vec<u8>, Vec<u8>)> = HashMap::new();

        // ---------------- rollout ----------------
        let t_roll = Instant::now();
        let mut fwd_calls = 0;
        loop {
            // observations grouped by (policy: 0 = learner, k + 1 = opponent k, width, height)
            let mut groups: HashMap<(usize, usize, usize), Vec<(usize, u8)>> = HashMap::new();
            for (gi, g) in games.iter().enumerate() {
                if g.env.done {
                    continue;
                }
                for p in 1..=g.env.n {
                    if g.env.alive(p) {
                        let pol = g.seats[p].map_or(0, |k| k + 1);
                        groups.entry((pol, g.env.width(), g.env.height())).or_default().push((gi, p as u8));
                    }
                }
            }
            if groups.is_empty() {
                break;
            }
            let mut dirs: Vec<Vec<u8>> = games.iter().map(|g| vec![0u8; g.env.width() * g.env.height()]).collect();
            let keep_turn: Vec<bool> = games.iter().map(|_| (rng.next() >> 11) as f64 / ((1u64 << 53) as f64) < keep).collect();
            let mut keys: Vec<_> = groups.keys().cloned().collect();
            keys.sort();
            // queue every group's forward pass before reading any result back, so the GPU runs
            // them in one go instead of waking up once per group
            let mut outputs = Vec::with_capacity(keys.len());
            for &(pol, w, h) in &keys {
                let obs = &groups[&(pol, w, h)];
                let mut x = Vec::with_capacity(obs.len() * w * h * nn::NUM_FEATURES);
                for &(gi, p) in obs {
                    let g = &games[gi];
                    let (own, st) = g.hist.last().unwrap();
                    x.extend(nn::features(w, h, own, st, &g.prod, p, g.env.turn));
                }
                let xt = Tensor::from_vec(x, (obs.len(), h, w, nn::NUM_FEATURES), &dev)?;
                fwd_calls += 1;
                outputs.push(if pol == 0 {
                    let (l, v) = model.forward_pv(&xt)?;
                    (l, Some(v))
                } else {
                    (opponents[pol - 1].model.forward(&xt)?, None)
                });
            }
            for (key, (lt, vt)) in keys.iter().zip(outputs) {
                let obs = &groups[key];
                let (pol, w, h) = *key;
                let n = w * h;
                let logits: Vec<f32> = lt.flatten_all()?.to_vec1()?;
                let values: Option<Vec<f32>> = vt.map(|v| v.to_vec1()).transpose()?;
                for (bi, &(gi, p)) in obs.iter().enumerate() {
                    let (own, st) = games[gi].hist.last().unwrap().clone();
                    let lg = &logits[bi * n * 5..(bi + 1) * n * 5];
                    let learner = pol == 0;
                    let mut actions = if learner { vec![0u8; n] } else { Vec::new() };
                    let mut logp = if learner { vec![0f32; n] } else { Vec::new() };
                    for i in 0..n {
                        if own[i] != p {
                            continue;
                        }
                        let l = &lg[i * 5..i * 5 + 5];
                        let d = if learner {
                            let (k, lp) = softmax_sample(l, temp as f32, &mut rng);
                            actions[i] = k as u8;
                            logp[i] = lp;
                            k
                        } else {
                            argmax(l)
                        };
                        dirs[gi][i] = d as u8;
                    }
                    if learner {
                        let ti = games[gi].traj[p as usize].unwrap();
                        let step = trajs[ti].values.len();
                        trajs[ti].values.push(values.as_ref().unwrap()[bi]);
                        trajs[ti].rewards.push(0.0);
                        if keep_turn[gi] {
                            let turn = games[gi].env.turn;
                            frames.entry((gi, turn)).or_insert((own, st));
                            kept.push(Kept { env: gi, turn, player: p, traj: ti, step, actions, old_logp: logp });
                        }
                    }
                }
            }
            // step every running game; shaping rewards; terminal rank rewards
            for (gi, g) in games.iter_mut().enumerate() {
                if g.env.done {
                    continue;
                }
                let moved: Vec<usize> = (1..=g.env.n).filter(|&p| g.env.alive(p)).collect();
                g.env.step(&dirs[gi]);
                g.hist.push(g.env.frame());
                let share = prod_share(&g.env);
                for &p in &moved {
                    if let Some(ti) = g.traj[p] {
                        let r = trajs[ti].rewards.last_mut().unwrap();
                        *r += shaping * (share[p] - g.share[p]) as f32;
                    }
                }
                g.share = share;
                if g.env.done {
                    let ranks = g.env.ranks();
                    let n = g.env.n;
                    for p in 1..=n {
                        if let Some(ti) = g.traj[p] {
                            let rr = if n > 1 { 1.0 - 2.0 * (ranks[p] - 1) as f32 / (n - 1) as f32 } else { 0.0 };
                            if let Some(r) = trajs[ti].rewards.last_mut() {
                                *r += rr;
                            }
                        }
                    }
                }
            }
        }
        let roll_secs = t_roll.elapsed().as_secs_f64();

        // ---------------- results of the iteration ----------------
        // games with both learner and frozen seats: per-seat win rate and rank score (1 first,
        // 0 last) of each side; equal policies would score the same. Solo games: final share of
        // the map's production.
        let (mut vs_games, mut turns) = (0, 0);
        let mut side = [[0f64; 3]; 2]; // [learner, frozen] x [seats, wins, rank score]
        let (mut solo_games, mut solo_share) = (0, 0.0);
        for g in &games {
            turns += g.env.turn;
            if g.env.n == 1 {
                solo_games += 1;
                solo_share += g.share[1];
            } else if g.seats.iter().skip(1).any(|s| s.is_some()) {
                vs_games += 1;
                let ranks = g.env.ranks();
                for p in 1..=g.env.n {
                    let s = &mut side[g.seats[p].is_some() as usize];
                    s[0] += 1.0;
                    s[1] += (ranks[p] == 1) as u8 as f64;
                    s[2] += 1.0 - (ranks[p] - 1) as f64 / (g.env.n - 1).max(1) as f64;
                }
            }
        }
        let rate = |s: [f64; 3], k: usize| s[k] / s[0].max(1.0);

        // ---------------- GAE ----------------
        for t in trajs.iter_mut() {
            let n = t.rewards.len();
            t.adv = vec![0.0; n];
            t.ret = vec![0.0; n];
            let mut next_adv = 0.0;
            for s in (0..n).rev() {
                let next_v = if s + 1 < n { t.values[s + 1] } else { 0.0 };
                let delta = t.rewards[s] + gamma * next_v - t.values[s];
                next_adv = delta + gamma * lam * next_adv;
                t.adv[s] = next_adv;
                t.ret[s] = next_adv + t.values[s];
            }
        }
        let advs: Vec<f32> = kept.iter().map(|k| trajs[k.traj].adv[k.step]).collect();
        let mean = advs.iter().sum::<f32>() / advs.len().max(1) as f32;
        let std = (advs.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / advs.len().max(1) as f32).sqrt().max(1e-6);
        // explained variance of the value head on kept frames
        let rets: Vec<f32> = kept.iter().map(|k| trajs[k.traj].ret[k.step]).collect();
        let vals: Vec<f32> = kept.iter().map(|k| trajs[k.traj].values[k.step]).collect();
        let var = |v: &[f32]| {
            let m = v.iter().sum::<f32>() / v.len().max(1) as f32;
            v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len().max(1) as f32
        };
        let resid: Vec<f32> = rets.iter().zip(&vals).map(|(r, v)| r - v).collect();
        let expl_var = 1.0 - var(&resid) / var(&rets).max(1e-8);

        // ---------------- local returns ----------------
        let mut local: HashMap<(usize, usize, u8), Vec<f32>> = HashMap::new();
        if local_w > 0.0 {
            let mut need: HashMap<(usize, u8), Vec<usize>> = HashMap::new();
            for k in &kept {
                need.entry((k.env, k.player)).or_default().push(k.turn);
            }
            for ((gi, p), turns_needed) in need {
                for (t, v) in local_returns(&games[gi], p, &turns_needed, &lc) {
                    local.insert((gi, t, p), v);
                }
            }
        }
        let ctx = Ctx { games: &games, kept: &kept, trajs: &trajs, frames: &frames, local: &local, mean, std };

        // ---------------- diagnostic: gradient signal to noise (--snr) ----------------
        // Policy gradient at ratio 1 (-sum(advantage x log-prob), frames weighted as in the update)
        // on the even and on the odd games, which are independent. If g = s + noise, the cosine
        // estimates |s|^2 / (|s|^2 + noise^2) for half the games; k times more games per half
        // divide the noise term by k. Run after the value warmup so the baselines are trained.
        if a.contains_key("snr") && it >= value_warmup {
            let vars: Vec<candle_core::Var> = varmap.data().lock().unwrap().iter()
                .filter(|(name, _)| !name.starts_with("value") && !name.starts_with("vmap")).map(|(_, v)| v.clone()).collect();
            for (label, lw, gw) in [("frame advantage", 0.0, 1.0), ("local + global", local_w.max(1.0), global_w), ("local only", local_w.max(1.0), 0.0)] {
                let mut grads: Vec<Vec<f32>> = Vec::new();
                for half in 0..2 {
                    let idx: Vec<usize> = (0..kept.len()).filter(|&ki| kept[ki].env % 2 == half).collect();
                    let mut acc: Vec<Vec<f32>> = vars.iter().map(|v| vec![0f32; v.elem_count()]).collect();
                    for mb in minibatches(&ctx, &idx, pixels, None) {
                        let b = build_batch(&ctx, &mb, &dev)?;
                        let (logits, _, vm) = model.forward_heads(&b.xt, true)?;
                        let aw = site_advantage(&b, &vm.unwrap().detach(), lw, gw)?;
                        let logp = candle_nn::ops::log_softmax(&(logits / temp)?, D::Minus1)?.gather(&b.act.unsqueeze(1)?, 1)?.squeeze(1)?;
                        let loss = ((logp * aw)?.sum_all()? / idx.len() as f64)?.neg()?;
                        let g = loss.backward()?;
                        for (vi, var) in vars.iter().enumerate() {
                            if let Some(t) = g.get(var) {
                                for (dst, src) in acc[vi].iter_mut().zip(t.flatten_all()?.to_vec1::<f32>()?) {
                                    *dst += src;
                                }
                            }
                        }
                    }
                    grads.push(acc.concat());
                }
                let dot: f64 = grads[0].iter().zip(&grads[1]).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
                let n0 = grads[0].iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                let n1 = grads[1].iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                let cos = dot / (n0 * n1).max(1e-30);
                say(format!(
                    "snr {label}: {} games per half, {} kept frames: cos(g_even, g_odd) = {:.4}; games per half for cos 0.5: {}",
                    n_games / 2, kept.len(), cos,
                    if cos > 0.0 { format!("{:.0}", (n_games / 2) as f64 * (1.0 - cos) / cos) } else { "unbounded (no signal measured)".into() }
                ));
            }
            return Ok(());
        }

        // ---------------- PPO update ----------------
        let t_upd = Instant::now();
        let policy_on = it >= value_warmup;
        let all: Vec<usize> = (0..kept.len()).collect();
        let mut stats = [0f64; 8]; // pg, value, entropy, approx kl, clip fraction, batches, teacher kl, value map
        for _ in 0..epochs {
            for mb in minibatches(&ctx, &all, pixels, Some(&mut rng)) {
                let b = build_batch(&ctx, &mb, &dev)?;
                let bs = b.bs as f64;
                let (logits, value, vm) = model.forward_heads(&b.xt, local_w > 0.0)?;
                let logp_all = candle_nn::ops::log_softmax(&(logits / temp)?, D::Minus1)?;
                // KL(policy || teacher) per site, the teacher being the starting network at the same
                // temperature: keeps the noisy policy gradient from walking away from it
                let tkl = if teacher_kl > 0.0 && policy_on {
                    let tl = candle_nn::ops::log_softmax(&(opponents[0].model.forward(&b.xt)?.detach() / temp)?, D::Minus1)?;
                    ((logp_all.exp()? * (&logp_all - &tl)?)?.sum(1)? * &b.sw)?.sum_all()?.affine(1.0 / bs, 0.0)?
                } else {
                    Tensor::zeros((), DType::F32, &dev)?
                };
                let zero = Tensor::zeros(b.lret.dims(), DType::F32, &dev)?;
                let vm_d = vm.as_ref().map(|v| v.detach()).unwrap_or(zero);
                let aw = site_advantage(&b, &vm_d, local_w, global_w)?;
                let logp = logp_all.gather(&b.act.unsqueeze(1)?, 1)?.squeeze(1)?;
                let ratio = (&logp - &b.old)?.exp()?;
                let pg1 = (&ratio * &aw)?;
                let pg2 = (ratio.clamp(1.0 - clip, 1.0 + clip)? * &aw)?;
                let pg = (pg1.minimum(&pg2)?.sum_all()? / bs)?.neg()?;
                let ent = ((logp_all.exp()? * &logp_all)?.sum(1)?.neg()? * &b.sw)?.sum_all()?.affine(1.0 / bs, 0.0)?;
                let vloss = (&value - &b.ret)?.sqr()?.mean_all()?;
                let mloss = match &vm {
                    Some(v) => ((v - &b.lret)?.sqr()? * &b.sw)?.sum_all()?.affine(1.0 / bs, 0.0)?,
                    None => Tensor::zeros((), DType::F32, &dev)?,
                };
                let vterm = ((&vloss + &mloss)? * vcoef)?;
                let loss = if policy_on {
                    (((&pg + &vterm)? - (&ent * ent_coef)?)? + (&tkl * teacher_kl)?)?
                } else {
                    vterm
                };
                opt.backward_step(&loss)?;
                let kl = (((&b.old - &logp)? * &b.sw)?.sum_all()? / bs)?;
                let clipped = ((&ratio - 1.0)?.abs()?.gt(clip as f64)?.to_dtype(DType::F32)? * &b.sw)?.sum_all()?;
                stats[0] += pg.to_scalar::<f32>()? as f64;
                stats[1] += vloss.to_scalar::<f32>()? as f64;
                stats[2] += ent.to_scalar::<f32>()? as f64;
                stats[3] += kl.to_scalar::<f32>()? as f64;
                stats[4] += clipped.to_scalar::<f32>()? as f64 / bs;
                stats[5] += 1.0;
                stats[6] += tkl.to_scalar::<f32>()? as f64;
                stats[7] += mloss.to_scalar::<f32>()? as f64;
            }
        }
        let nb = stats[5].max(1.0);
        let upd_secs = t_upd.elapsed().as_secs_f64();
        say(format!(
            "iter {:4}  win {:.3} vs frozen {:.3}  rank score {:.3} vs {:.3}  ({} mixed games)  pg {:+.4}  value {:.4}  expl var {:.3}  entropy {:.4}  kl {:.4}  clip {:.3}  teacher kl {:.4}  frames {}  turns {}  rollout {:.0}s ({} fwd)  update {:.0}s  total {:.0}s  vmap {:.4}  solo {:.4} ({} games){}",
            it + 1, rate(side[0], 1), rate(side[1], 1), rate(side[0], 2), rate(side[1], 2), vs_games,
            stats[0] / nb, stats[1] / nb, expl_var, stats[2] / nb, stats[3] / nb, stats[4] / nb, stats[6] / nb,
            kept.len(), turns, roll_secs, fwd_calls, upd_secs, t_iter.elapsed().as_secs_f64(),
            stats[7] / nb, solo_share / solo_games.max(1) as f64, solo_games,
            if policy_on { "" } else { "  (value warmup)" }
        ));

        if (it + 1) % save_every == 0 || it + 1 == iters {
            varmap.save(run.join("model.safetensors"))?;
            std::fs::write(run.join("model.hnn"), model.export()?).unwrap();
            std::fs::write(run.join(format!("iter{:04}.hnn", it + 1)), model.export()?).unwrap();
        }
        if (it + 1) % snapshot_every == 0 {
            varmap.save(run.join("snapshot.safetensors"))?;
            let p = load_policy(&run.join("snapshot.safetensors"), channels, &dil, &dev, format!("iter{}", it + 1))?;
            if opponents.len() > max_snapshots {
                opponents.remove(1); // keep the starting network, drop the oldest snapshot
            }
            opponents.push(p);
            say(format!("opponents: {:?}", opponents.iter().map(|o| o.name.clone()).collect::<Vec<_>>()));
        }
    }
    Ok(())
}
