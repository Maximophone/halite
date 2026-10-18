//! Reinforcement learning: PPO self-play on in-process games (sim.rs), starting from a cloned net.
//!
//!     halite-rl rl --init rl/runs/bc3/model.safetensors --run rl/runs/ppo1 [--iters 300] ...
//!
//! Each iteration plays `--games` games at once. The games come in `--types` (map side, players)
//! groups so that every turn needs only a few batched forward passes on the GPU. One seat per game
//! is the learner; every other seat is the learner too with probability `--self-share`, otherwise
//! a frozen opponent (the starting network, plus snapshots of the learner added every
//! `--snapshot-every` iterations) playing its most likely move, like the released bot.
//! The learner samples its moves from softmax(logits / --temp) (0.5: close to the bot's argmax;
//! the same temperature is used in the update, so the ratio is exact). Reward: final rank (+1 first .. -1 last) at the seat's last
//! turn, plus `--shaping` x the change of its share of all player-owned production each turn
//! (a potential, so it does not change what is optimal). GAE, then PPO with a per-site clipped
//! ratio, each frame weighing the same as in cloning, a value loss and an entropy bonus. The
//! first `--value-warmup` iterations train only the value head (which reads a detached trunk, so it
//! never moves the policy).
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
}

fn prod_share(env: &Env) -> Vec<f64> {
    let p = env.production_owned();
    let total: i64 = p[1..].iter().sum();
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

pub fn run() -> Result<()> {
    let a = crate::args();
    let get = |k: &str, d: &str| a.get(k).cloned().unwrap_or(d.to_string());
    let num = |k: &str, d: &str| get(k, d).parse::<f64>().unwrap();
    let run = PathBuf::from(get("run", "rl/runs/ppo"));
    std::fs::create_dir_all(&run).unwrap();
    let init = PathBuf::from(get("init", "rl/runs/bc3/model.safetensors"));
    let iters = num("iters", "300") as usize;
    let n_games = num("games", "48") as usize;
    let n_types = num("types", "8") as usize;
    let keep = num("keep", "0.25");
    let epochs = num("epochs", "2") as usize;
    let lr = num("lr", "1e-4");
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
            .map(|_| ([20usize, 25, 30, 35, 40, 45, 50][rng.below(7)], [2usize, 2, 3, 4, 4, 5, 6][rng.below(7)]))
            .collect();
        for g in 0..n_games {
            let (side, n) = types[g % n_types];
            let env = Env::new(rng.next() as u32, side, side, n);
            let prod = env.productions();
            let learner_seat = 1 + rng.below(n);
            let mut seats = vec![None; n + 1];
            for (p, seat) in seats.iter_mut().enumerate().skip(1) {
                if p != learner_seat && (rng.next() >> 11) as f64 / (1u64 << 53) as f64 >= self_share {
                    *seat = Some(rng.below(opponents.len()));
                }
            }
            let share = prod_share(&env);
            games.push(Game { env, prod, seats, traj: vec![None; n + 1], share });
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
            let frame_cache: Vec<Option<(Vec<u8>, Vec<u8>)>> =
                games.iter().map(|g| if g.env.done { None } else { Some(g.env.frame()) }).collect();
            // queue every group's forward pass before reading any result back, so the GPU runs
            // them in one go instead of waking up once per group
            let mut outputs = Vec::with_capacity(keys.len());
            for &(pol, w, h) in &keys {
                let obs = &groups[&(pol, w, h)];
                let mut x = Vec::with_capacity(obs.len() * w * h * nn::NUM_FEATURES);
                for &(gi, p) in obs {
                    let (own, st) = frame_cache[gi].as_ref().unwrap();
                    x.extend(nn::features(w, h, own, st, &games[gi].prod, p, games[gi].env.turn));
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
                    let (own, st) = frame_cache[gi].as_ref().unwrap();
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
                            frames.entry((gi, turn)).or_insert_with(|| (own.clone(), st.clone()));
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
        // 0 last) of each side; equal policies would score the same
        let (mut vs_games, mut turns) = (0, 0);
        let mut side = [[0f64; 3]; 2]; // [learner, frozen] x [seats, wins, rank score]
        for g in &games {
            turns += g.env.turn;
            if g.seats.iter().skip(1).any(|s| s.is_some()) {
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

        // ---------------- PPO update ----------------
        let t_upd = Instant::now();
        let policy_on = it >= value_warmup;
        let mut by_size: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (ki, k) in kept.iter().enumerate() {
            let e = &games[k.env].env;
            by_size.entry((e.width(), e.height())).or_default().push(ki);
        }
        let mut stats = [0f64; 6]; // pg, value, entropy, approx kl, clip fraction, batches
        for _ in 0..epochs {
            let mut batches: Vec<Vec<usize>> = Vec::new();
            for ((w, h), v) in by_size.iter() {
                let mut v = v.clone();
                for i in (1..v.len()).rev() {
                    v.swap(i, rng.below(i + 1));
                }
                let per = (pixels / (w * h)).max(1);
                batches.extend(v.chunks(per).map(|c| c.to_vec()));
            }
            for i in (1..batches.len()).rev() {
                batches.swap(i, rng.below(i + 1));
            }
            for b in &batches {
                let e0 = &games[kept[b[0]].env].env;
                let (w, h) = (e0.width(), e0.height());
                let n = w * h;
                let bs = b.len();
                let mut x = Vec::with_capacity(bs * n * nn::NUM_FEATURES);
                let mut act = Vec::with_capacity(bs * n);
                let mut old = Vec::with_capacity(bs * n);
                let mut aw = Vec::with_capacity(bs * n); // advantage x site weight
                let mut sw = Vec::with_capacity(bs * n); // site weight (1 / sites owned)
                let mut ret = Vec::with_capacity(bs);
                for &ki in b {
                    let k = &kept[ki];
                    let (own, st) = &frames[&(k.env, k.turn)];
                    x.extend(nn::features(w, h, own, st, &games[k.env].prod, k.player, k.turn));
                    let owned = own.iter().filter(|&&o| o == k.player).count().max(1) as f32;
                    let adv = (trajs[k.traj].adv[k.step] - mean) / std;
                    for i in 0..n {
                        let m = (own[i] == k.player) as u8 as f32 / owned;
                        act.push(k.actions[i] as u32);
                        old.push(k.old_logp[i]);
                        aw.push(adv * m);
                        sw.push(m);
                    }
                    ret.push(trajs[k.traj].ret[k.step]);
                }
                let xt = Tensor::from_vec(x, (bs, h, w, nn::NUM_FEATURES), &dev)?;
                let act = Tensor::from_vec(act, bs * n, &dev)?;
                let old = Tensor::from_vec(old, bs * n, &dev)?;
                let aw = Tensor::from_vec(aw, bs * n, &dev)?;
                let sw = Tensor::from_vec(sw, bs * n, &dev)?;
                let ret = Tensor::from_vec(ret, bs, &dev)?;
                let (logits, value) = model.forward_pv(&xt)?;
                let logp_all = candle_nn::ops::log_softmax(&(logits / temp)?, D::Minus1)?;
                let logp = logp_all.gather(&act.unsqueeze(1)?, 1)?.squeeze(1)?;
                let ratio = (&logp - &old)?.exp()?;
                let pg1 = (&ratio * &aw)?;
                let pg2 = (ratio.clamp(1.0 - clip, 1.0 + clip)? * &aw)?;
                let pg = (pg1.minimum(&pg2)?.sum_all()? / bs as f64)?.neg()?;
                let ent = ((logp_all.exp()? * &logp_all)?.sum(1)?.neg()? * &sw)?.sum_all()?.affine(1.0 / bs as f64, 0.0)?;
                let vloss = (&value - &ret)?.sqr()?.mean_all()?;
                let loss = if policy_on {
                    ((&pg + (&vloss * vcoef)?)? - (&ent * ent_coef)?)?
                } else {
                    (&vloss * vcoef)?
                };
                opt.backward_step(&loss)?;
                let kl = (((&old - &logp)? * &sw)?.sum_all()? / bs as f64)?;
                let clipped = ((&ratio - 1.0)?.abs()?.gt(clip as f64)?.to_dtype(DType::F32)? * &sw)?.sum_all()?;
                stats[0] += pg.to_scalar::<f32>()? as f64;
                stats[1] += vloss.to_scalar::<f32>()? as f64;
                stats[2] += ent.to_scalar::<f32>()? as f64;
                stats[3] += kl.to_scalar::<f32>()? as f64;
                stats[4] += clipped.to_scalar::<f32>()? as f64 / bs as f64;
                stats[5] += 1.0;
            }
        }
        let nb = stats[5].max(1.0);
        let upd_secs = t_upd.elapsed().as_secs_f64();
        say(format!(
            "iter {:4}  win {:.3} vs frozen {:.3}  rank score {:.3} vs {:.3}  ({} mixed games)  pg {:+.4}  value {:.4}  expl var {:.3}  entropy {:.4}  kl {:.4}  clip {:.3}  frames {}  turns {}  rollout {:.0}s ({} fwd)  update {:.0}s  total {:.0}s{}",
            it + 1, rate(side[0], 1), rate(side[1], 1), rate(side[0], 2), rate(side[1], 2), vs_games,
            stats[0] / nb, stats[1] / nb, expl_var, stats[2] / nb, stats[3] / nb, stats[4] / nb,
            kept.len(), turns, roll_secs, fwd_calls, upd_secs, t_iter.elapsed().as_secs_f64(),
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
